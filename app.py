from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import pandas as pd
import csv
import random
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# --- GÜVENLİK AYARLARI ---
app.secret_key = 'cok_gizli_ve_rastgele_bir_anahtar'
ADMIN_SIFRESI = '1811'

# --- DOSYA AYARLARI ---
CSV_DOSYASI = r'C:\Users\pc-one\OneDrive\Masaüstü\PROJE\sanatcisöz.csv'
MODEL_DOSYASI = 'model.pkl'
VEC_DOSYASI = 'vectorizer.pkl'

# Global değişkenler
model = None
vectorizer = None

def modelleri_yukle():
    global model, vectorizer
    if os.path.exists(MODEL_DOSYASI) and os.path.exists(VEC_DOSYASI):
        model = joblib.load(MODEL_DOSYASI)
        vectorizer = joblib.load(VEC_DOSYASI)
        return True
    return False

MODEL_YUKLENDI = modelleri_yukle()

def metni_temizle(metin):
    temiz = str(metin).lower()
    temiz = temiz.replace('\n', ' ').replace('\r', ' ')
    temiz = re.sub(r'[^\w\s]', '', temiz) 
    temiz = " ".join(temiz.split())
    return temiz

def verileri_grupla(df, grup_boyutu=5):
    """Verileri 5'erli birleştirip paragraf yapar."""
    yeni_sozler = []
    yeni_sanatcilar = []

    sanatcilar = df['sanatci'].unique()
    
    for sanatci in sanatcilar:
        sanatci_df = df[df['sanatci'] == sanatci]
        sozler = sanatci_df['soz'].tolist()
        
        for i in range(0, len(sozler), grup_boyutu):
            grup = " ".join(sozler[i:i+grup_boyutu])
            if len(grup) > 20: 
                yeni_sozler.append(grup)
                yeni_sanatcilar.append(sanatci)
                
    return pd.DataFrame({'sanatci': yeni_sanatcilar, 'soz': yeni_sozler})

def get_mevcut_sanatcilar():
    try:
        df = pd.read_csv(CSV_DOSYASI, usecols=[0], encoding='utf-8', on_bad_lines='skip')
        # İsimleri düzeltip döndür
        isimler = df.iloc[:, 0].astype(str).str.strip().str.title().unique().tolist()
        return isimler
    except:
        return []

def get_mevcut_sozler_seti():
    try:
        df = pd.read_csv(CSV_DOSYASI, usecols=[1], encoding='utf-8', on_bad_lines='skip')
        return set(df.iloc[:, 0].astype(str).tolist())
    except:
        return set()

# --- ROTALAR ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    hata = None
    if request.method == 'POST':
        sifre = request.form['sifre']
        if sifre == ADMIN_SIFRESI:
            session['giris_yapildi'] = True
            return redirect(url_for('ekle'))
        else:
            hata = "❌ Yanlış Şifre!"
    return render_template('login.html', hata=hata)

@app.route('/logout')
def logout():
    session.pop('giris_yapildi', None)
    return redirect(url_for('index'))

def neden_bunu_secti(model, vectorizer, metin, sanatci_index):
    """
    Modelin neden bu kararı verdiğini matematiksel olarak açıklar.
    O sanatçının puanını en çok artıran kelimeleri bulur.
    """
    try:
        # 1. Metni vektöre çevir
        vektor = vectorizer.transform([metin])
        feature_names = vectorizer.get_feature_names_out()
        
        # 2. Sadece metinde geçen kelimelere odaklan
        feature_index = vektor.nonzero()[1]
        
        # 3. O sanatçının katsayılarını (coefficients) al
        # model.coef_ matrisi [Sanatçı Sayısı, Kelime Sayısı] şeklindedir
        coefs = model.coef_[sanatci_index]
        
        kelime_skorlari = []
        for idx in feature_index:
            weight = coefs[idx] # Kelimenin o sanatçıdaki ağırlığı
            tfidf_val = vektor[0, idx] # Kelimenin metindeki gücü
            
            # Eğer ağırlık pozitifse (yani bu kelime o sanatçıya aitse)
            if weight > 0:
                score = weight * tfidf_val
                kelime_skorlari.append((feature_names[idx], score))
        
        # 4. En yüksek puanlı 5 kelimeyi seç
        kelime_skorlari.sort(key=lambda x: x[1], reverse=True)
        return [k[0] for k in kelime_skorlari[:5]]
    except Exception as e:
        print(f"İpucu hatası: {e}")
        return []

@app.route('/tahmin', methods=['GET', 'POST'])
def tahmin():
    if not MODEL_YUKLENDI:
        return render_template('tahmin.html', hata="Model henüz eğitilmedi! Lütfen önce eğitimi başlatın.")

    top3_sonuc = None
    soz = ""

    if request.method == 'POST':
        soz = request.form['sarki_sozu']
        if soz.strip():
            try:
                temiz_input = metni_temizle(soz)
                vektor = vectorizer.transform([temiz_input])
                probs = model.predict_proba(vektor)[0]
                classes = model.classes_
                
                # Olasılıkları sırala
                sirali_liste = sorted(zip(classes, probs, range(len(classes))), key=lambda x: x[1], reverse=True)
                
                top3_sonuc = []
                for i, (sanatci, puan, index_no) in enumerate(sirali_liste[:3]):
                    
                    # tahminler için "Neden?" analizi yap
                    ipucu_kelimeler = []
                     
                    ipucu_kelimeler = neden_bunu_secti(model, vectorizer, temiz_input, index_no)
                    

                    top3_sonuc.append({
                        'sanatci': sanatci,
                        'oran': f"{puan * 100:.2f}",
                        'ham_puan': puan * 100,
                        'ipucu': ipucu_kelimeler # İpuçlarını buraya ekledik
                    })
                
            except Exception as e:
                return f"Hata: {e}"
                
    return render_template('tahmin.html', top3=top3_sonuc, soz=soz)

@app.route('/sanatcilar')
def sanatcilar():
    mevcut = get_mevcut_sanatcilar()
    return render_template('sanatcilar.html', sanatcilar=sorted(mevcut))

@app.route('/ekle', methods=['GET', 'POST'])
def ekle():
    if not session.get('giris_yapildi'):
        return redirect(url_for('login'))

    mesaj = request.args.get('mesaj')
    
    if request.method == 'POST':
        # İsimleri otomatik düzelt (.title())
        yeni_sanatci = request.form['sanatci_adi'].strip().title()
        gelen_liste = request.form.getlist('coklu_sozler[]')
        
        mevcut_sanatcilar = get_mevcut_sanatcilar()
        sanatci_var_mi = yeni_sanatci in mevcut_sanatcilar

        try:
            veritabani_sozleri = get_mevcut_sozler_seti()
            eklenecekler = []
            
            for ham_soz in gelen_liste:
                if len(ham_soz.strip()) > 5:
                    temiz = metni_temizle(ham_soz)
                    if temiz not in veritabani_sozleri:
                        eklenecekler.append(temiz)
                        veritabani_sozleri.add(temiz) 

            sarki_sayisi = len(eklenecekler)
            
            # Veri ekleme limiti
            limit = 5 
            if sanatci_var_mi or sarki_sayisi >= limit:
                if sarki_sayisi > 0:
                    with open(CSV_DOSYASI, mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        for sarki in eklenecekler:
                            writer.writerow([yeni_sanatci, sarki])
                    mesaj = f"✅ BAŞARILI: {yeni_sanatci} için {sarki_sayisi} satır eklendi."
                else:
                    mesaj = "⚠️ Eklemeye çalıştığınız veriler zaten vardı."
            else:
                mesaj = f"❌ YENİ SANATÇI İÇİN EN AZ {limit} SATIR GİRİN."

        except Exception as e:
            mesaj = f"❌ Hata: {e}"

    return render_template('ekle.html', mesaj=mesaj)

# --- OYUN MODU ROTASI ---
@app.route('/oyun')
def oyun():
    try:
        # Veriyi oku
        df = pd.read_csv(CSV_DOSYASI, encoding='utf-8', on_bad_lines='skip')
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ['sanatci', 'soz']
        
        # Temizlik (Kısa ve anlamsız satırları oyuna sokma)
        df = df.dropna(subset=['soz'])
        df['soz'] = df['soz'].astype(str)
        df['sanatci'] = df['sanatci'].str.strip().str.title()
        
        # Sadece uzunluğu 30 karakterden fazla olan anlamlı sözleri al
        oyun_df = df[df['soz'].str.len() > 30]
        
        if len(oyun_df) < 10:
            return "Oyun oynamak için veritabanında yeterli uzunlukta şarkı sözü yok!"

        # 1. Rastgele bir satır seç (Soru)
        rastgele_satir = oyun_df.sample(1).iloc[0]
        soru_soz = rastgele_satir['soz']
        dogru_cevap = rastgele_satir['sanatci']
        
        # 2. Yanlış şıkları belirle
        tum_sanatcilar = df['sanatci'].unique().tolist()
        
        # Doğru cevabı listeden çıkar ki yanlışlıkla seçmeyelim
        if dogru_cevap in tum_sanatcilar:
            tum_sanatcilar.remove(dogru_cevap)
            
        # 3 tane yanlış cevap seç
        siklar = random.sample(tum_sanatcilar, 3)
        
        # Doğru cevabı ekle ve karıştır
        siklar.append(dogru_cevap)
        random.shuffle(siklar)
        
        return render_template('oyun.html', soru=soru_soz, siklar=siklar, dogru=dogru_cevap)
        
    except Exception as e:
        return f"Oyun Hatası: {e}"

# --- FİNAL STABİL EĞİTİM (RAPORLU) ---

@app.route('/egit', methods=['POST'])
def egit():
    if not session.get('giris_yapildi'):
        return redirect(url_for('login'))

    global MODEL_YUKLENDI, model, vectorizer
    try:
        df = pd.read_csv(CSV_DOSYASI, encoding='utf-8', on_bad_lines='skip')
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ['sanatci', 'soz']
        
        # Temizlik
        df = df.dropna(subset=['soz'])
        df['soz'] = df['soz'].astype(str)
        df['sanatci'] = df['sanatci'].str.strip().str.title()
        df['soz'] = df['soz'].apply(metni_temizle)
        df = df[df['soz'].str.len() > 10]

        print("\n" + "="*50)
        print("🧪 GRID SEARCH + HATA ANALİZİ BAŞLIYOR...")
        
        # Grup Boyutunu Ayarla
        toplam_satir = len(df)
        dinamik_grup = 6 if toplam_satir > 3000 else 4
        
        df_gruplu = verileri_grupla(df, grup_boyutu=dinamik_grup)
        
        sanatci_sayilari = df_gruplu['sanatci'].value_counts()
        yetersiz_sanatcilar = sanatci_sayilari[sanatci_sayilari < 3].index
        if len(yetersiz_sanatcilar) > 0:
            df_gruplu = df_gruplu[~df_gruplu['sanatci'].isin(yetersiz_sanatcilar)]
        
        print(f"📊 İşlenen Paragraf Sayısı: {len(df_gruplu)}")

        X = df_gruplu['soz']
        y = df_gruplu['sanatci']

        # TEST
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 1. Vektörleştirme
        vec = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)
        
        # 2. GRID SEARCH 
        parametreler = {
            'alpha': [1e-3, 1e-4, 1e-5],
            'penalty': ['l2', 'elasticnet'],
            'loss': ['modified_huber', 'log_loss'],
        }
        
        sgd = SGDClassifier(random_state=42, class_weight='balanced', max_iter=5000)
        
        # n_jobs=1 (Windows hatası vermesin diye)
        grid_search = GridSearchCV(sgd, parametreler, cv=3, n_jobs=1, verbose=1)
        grid_search.fit(X_train_vec, y_train)
        
        en_iyi_model = grid_search.best_estimator_
        
        print(f"💎 EN İYİ AYARLAR: {grid_search.best_params_}")
        
        preds = en_iyi_model.predict(X_test_vec)
        dogruluk = accuracy_score(y_test, preds) * 100
        
        # --- KARIŞIKLIK RAPORU (KİM KİMİNLE KARIŞIYOR?) ---
        print("\n" + "!"*50)
        print("🕵️ HATA ANALİZ RAPORU (CONFUSION REPORT):")
        print("Aşağıdaki liste, modelin en çok karıştırdığı ikilileri gösterir.")
        print("-" * 50)
        
        y_test_list = y_test.tolist() # Listeye çevir ki index hatası olmasın
        hatalar = []
        
        # Hataları topla
        for i in range(len(y_test_list)):
            gercek = y_test_list[i]
            tahmin_edilen = preds[i]
            
            if gercek != tahmin_edilen:
                hatalar.append(f"{gercek} -> {tahmin_edilen} sanıldı")
        
        # Hataları say ve en çok yapılanları yazdır
        hata_sayaci = Counter(hatalar)
        en_cok_hatalar = hata_sayaci.most_common(10) # En sık yapılan 10 hata
        
        if not en_cok_hatalar:
            print("🎉 MÜKEMMEL! Hiç hata yok (veya test seti çok küçük).")
        else:
            print(f"{'GERÇEK SANATÇI':<20} | {'YANLIŞ TAHMİN':<20} | {'ADET'}")
            print("-" * 55)
            for hata_metni, adet in en_cok_hatalar:
                parcali = hata_metni.split(" -> ")
                gercek_kisi = parcali[0]
                yanlis_kisi = parcali[1].replace(" sanıldı", "")
                print(f"{gercek_kisi:<20} | {yanlis_kisi:<20} | {adet} Kere")
            print("-" * 55)
            print("💡 İPUCU: Yukarıdaki sanatçıların birbirine benzeyen şarkılarını incele.")
            print("   Eğer 'Ceza -> Sagopa' hatası çoksa, ikisine de daha karakteristik (farklı) şarkılar ekle.")

        print("!"*50 + "\n")

        # FİNAL KAYIT
        final_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
        X_tfidf = final_vec.fit_transform(X)
        
        final_clf = SGDClassifier(
            **grid_search.best_params_, 
            random_state=42, 
            max_iter=5000, 
            class_weight='balanced'
        )
        final_clf.fit(X_tfidf, y)

        joblib.dump(final_clf, MODEL_DOSYASI)
        joblib.dump(final_vec, VEC_DOSYASI)
        
        model, vectorizer = final_clf, final_vec
        MODEL_YUKLENDI = True
        
        return redirect(url_for('ekle', mesaj=f"✅ Analiz Tamamlandı! Doğruluk: %{dogruluk:.2f} (Hatalar Terminalde)"))
    
    except Exception as e:
        print(f"❌ HATA: {e}")
        return redirect(url_for('ekle', mesaj=f"❌ Hata: {e}"))
    
if __name__ == '__main__':
    
    app.run(debug=True, host='0.0.0.0', port=5000)
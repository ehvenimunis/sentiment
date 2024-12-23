import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# CSV dosyasını oku
df = pd.read_csv('magaza_yorumlari_duygu_analizi.csv', delimiter=',', encoding='utf-16')

# Veri çerçevesini düzenle
df.columns = ["gorus", "durum"]  # Sütun adlarını dosyaya göre düzenleyin.

# Eksik değerleri kontrol etme
print("Eksik değerler:\n", df.isnull().sum())

# Eksik değerleri (NaN) temizleme - Bu örnekte NaN içeren satırlar kaldırılıyor
df = df.dropna(subset=["gorus", "durum"])

# Özellikleri (gorus) ve hedef değişkeni (durum) ayırma
X = df["gorus"]
y = df["durum"]

# Metinleri sayısal verilere dönüştürme (Bag of Words yöntemi)
vectorizer = CountVectorizer(stop_words='english')  # Stop words kullanarak daha doğru modelleme
X_vectorized = vectorizer.fit_transform(X)

# Veriyi eğitim ve test setine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Naive Bayes modelini eğitme
model = MultinomialNB()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları değerlendirme
print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))

# Kullanıcı girdisi için tahmin yapma fonksiyonu
def duygu_tahmin_et(metin):
    metin_vectorized = vectorizer.transform([metin])
    tahmin = model.predict(metin_vectorized)
    return tahmin[0]

# Örnek kullanıcı girdisi
kullanici_metin = "Bu restoranın yemekleri çok kötü."
tahmin = duygu_tahmin_et(kullanici_metin)
print(f"'{kullanici_metin}' için tahmin edilen duygu: {tahmin}")

This project demonstrates a sentiment analysis model using Naive Bayes classifier. It uses customer reviews and sentiment labels (positive, neutral, negative) to predict the sentiment of a given sentence. The process involves reading a CSV file, preparing the text data, and training a machine learning model to classify sentiments.
Requirements

    Python 3.x
    Pandas
    Scikit-learn

You can install the required packages using pip:
pip install pandas scikit-learn

Dataset

The dataset is assumed to be in CSV format, containing two columns:

    gorus: Customer reviews or sentences.
    durum: Sentiment labels (positive, negative, neutral).

Example of the dataset format:
gorus, durum
"Bu restoranın yemekleri çok kötü.", "Olumsuz"
"Yemekler harikaydı, teşekkürler!", "Olumlu"

Steps
1. Data Loading:

The data is loaded from a CSV file (magaza_yorumlari_duygu_analizi.csv) using Pandas. The file is expected to have two columns: gorus (review) and durum (sentiment). The CSV is read with UTF-16 encoding.
2. Data Preprocessing:

    Missing values are checked and removed using the dropna() method to ensure that the model is trained on complete data.
    The column names are updated to make them more descriptive (gorus and durum).

3. Text Vectorization:

The CountVectorizer from Scikit-learn is used to convert text data into numerical vectors. This process uses the "Bag of Words" model to create a sparse matrix of word frequencies. It also removes common "stop words" (words that do not add much meaning, like "and", "the", etc.).
4. Model Training:

The data is split into training and testing sets using train_test_split(). 80% of the data is used for training, while 20% is used for testing.

A Multinomial Naive Bayes classifier (MultinomialNB()) is used for sentiment classification. This model is well-suited for text classification problems, especially when features (words) are independent.
5. Model Evaluation:

The model's performance is evaluated using two metrics:

    Accuracy Score: This metric indicates the proportion of correct predictions.
    Classification Report: It provides a detailed performance report, including precision, recall, F1-score for each sentiment class.

6. User Input Prediction:

A function duygu_tahmin_et() is defined to predict the sentiment of new user input. The function takes a sentence, vectorizes it using the trained vectorizer, and predicts the sentiment label using the trained Naive Bayes model.
Example:

For example, the input sentence "Bu restoranın yemekleri çok kötü." will be classified as "Olumsuz" (negative sentiment).
kullanici_metin = "Bu restoranın yemekleri çok kötü."
tahmin = duygu_tahmin_et(kullanici_metin)
print(f"'{kullanici_metin}' için tahmin edilen duygu: {tahmin}")

Output

The output will display the following:

    Accuracy Score: The percentage of correct classifications on the test set.
    Classification Report: Detailed metrics for each sentiment class.
    Prediction: Sentiment prediction for the user input sentence.

Example:
Doğruluk Oranı: 0.85

Sınıflandırma Raporu:
               precision    recall  f1-score   support

        Olumlu       0.89      0.85      0.87        50
        Olumsuz      0.80      0.84      0.82        40
        Tarafsız     0.90      0.92      0.91        60

    accuracy                           0.85       150
   macro avg       0.86      0.87      0.86       150
weighted avg       0.86      0.85      0.86       150

'Bu restoranın yemekleri çok kötü.' için tahmin edilen duygu: Olumsuz

Conclusion

This project shows how to perform sentiment analysis on customer reviews using Naive Bayes. You can further improve the model by using more sophisticated techniques like TF-IDF, and fine-tuning other parameters of the model.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import lime
import lime.lime_text
import shap
import json

class DravidianFakeNewsDetector:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        self.tokenizer = Tokenizer(num_words=5000)
        self.label_encoder = LabelEncoder()
        self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=50, random_state=42)
        self.xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.01, random_state=42)
        self.lstm_model = None
        # Using XLM-RoBERTa for multilingual support
        self.bert_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.bert_model = AutoModel.from_pretrained("xlm-roberta-base")
        
    def preprocess_data(self, df):
        # Combine text and language information
        df['combined_text'] = df['language'] + ' ' + df['news_text']
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['label'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['combined_text'], y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_bert_embeddings(self, texts):
        embeddings = []
        batch_size = 32
        
        # Convert texts to list if it's not already
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Ensure batch is a list of strings
            batch = [str(text) for text in batch]
            
            inputs = self.bert_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True
            )
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            # Use attention mask for better embeddings
            attention_mask = inputs['attention_mask']
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings)

    def extract_features(self, X_train, X_test, y_train):
        # TF-IDF features
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        # Feature selection
        n_features = min(2000, X_train_tfidf.shape[1])
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_train_tfidf = selector.fit_transform(X_train_tfidf, y_train)
        X_test_tfidf = selector.transform(X_test_tfidf)
        
        print("Extracting BERT embeddings...")
        X_train_bert = self.get_bert_embeddings(X_train)
        X_test_bert = self.get_bert_embeddings(X_test)
        
        # Combine features
        X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_bert])
        X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_bert])
        
        return X_train_combined, X_test_combined
    
    def build_lstm_model(self, input_dim):
        # Create a simpler LSTM model that works with our feature input
        self.lstm_model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        self.lstm_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train_models(self, X_train, y_train):
        print("Training Random Forest...")
        self.rf_model.fit(X_train, y_train)
        
        print("Training XGBoost...")
        self.xgb_model.fit(X_train, y_train)
        
        print("Training LSTM...")
        self.build_lstm_model(X_train.shape[1])
        self.lstm_model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
    
    def ensemble_predict(self, X):
        # Get predictions from each model
        rf_pred = self.rf_model.predict_proba(X)[:, 1]
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
        lstm_pred = self.lstm_model.predict(X).flatten()
        
        # Weighted ensemble
        weights = [0.4, 0.3, 0.3]  # Weights for RF, XGBoost, and LSTM respectively
        ensemble_pred = (weights[0] * rf_pred + 
                        weights[1] * xgb_pred + 
                        weights[2] * lstm_pred)
        
        return (ensemble_pred > 0.5).astype(int)
    
    def explain_prediction(self, text, language):
        # Combine text and language information
        combined_text = language + ' ' + text
        
        # Extract features
        tfidf_features = self.tfidf_vectorizer.transform([combined_text])
        bert_features = self.get_bert_embeddings([combined_text])
        X = np.hstack([tfidf_features.toarray(), bert_features])
        
        # Get model predictions
        prediction = self.ensemble_predict(X)[0]
        
        # Initialize LIME explainer
        explainer = lime.lime_text.LimeTextExplainer(class_names=['Real', 'Fake'])
        
        # Create a prediction function for LIME
        def predict_proba(texts):
            # Process each text
            processed_texts = [language + ' ' + t for t in texts]
            tfidf = self.tfidf_vectorizer.transform(processed_texts)
            bert = self.get_bert_embeddings(processed_texts)
            X_combined = np.hstack([tfidf.toarray(), bert])
            
            # Get predictions from each model
            rf_pred = self.rf_model.predict_proba(X_combined)
            xgb_pred = self.xgb_model.predict_proba(X_combined)
            lstm_pred = np.stack([1-self.lstm_model.predict(X_combined).flatten(), 
                                 self.lstm_model.predict(X_combined).flatten()], axis=1)
            
            # Weighted ensemble
            weights = [0.4, 0.3, 0.3]
            return weights[0]*rf_pred + weights[1]*xgb_pred + weights[2]*lstm_pred
        
        # Generate LIME explanation
        lime_exp = explainer.explain_instance(text, predict_proba, num_features=10)
        
        # Get LIME explanation as a dictionary
        lime_explanation = {
            'text': text,
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': float(predict_proba([text])[0][1]),  # Probability of being fake
            'word_importance': [
                {'word': word, 'importance': float(importance)}
                for word, importance in lime_exp.as_list()
            ]
        }
        
        # Generate SHAP values for feature importance
        explainer = shap.KernelExplainer(predict_proba, 
                                        shap.sample(X, 100),  # Use a background dataset
                                        link="logit")
        shap_values = explainer.shap_values(X)
        
        # Get feature names
        feature_names = list(self.tfidf_vectorizer.get_feature_names_out()) + \
                       [f'bert_dim_{i}' for i in range(bert_features.shape[1])]
        
        # Get top SHAP features
        shap_importance = {
            'feature_importance': [
                {'feature': feature_names[i],
                 'importance': float(abs(shap_values[1][0][i]))}
                for i in np.argsort(abs(shap_values[1][0]))[-10:]
            ]
        }
        
        return {
            'lime_explanation': lime_explanation,
            'shap_explanation': shap_importance
        }
    
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()

def main():
    print("Loading data...")
    # Load data
    df = pd.read_csv('balanced_dravidian_fake_news_dataset.csv')
    
    print("Initializing detector...")
    # Initialize detector
    detector = DravidianFakeNewsDetector()
    
    print("Preprocessing data...")
    # Preprocess data
    X_train, X_test, y_train, y_test = detector.preprocess_data(df)
    
    print("Extracting features...")
    # Extract features
    X_train_combined, X_test_combined = detector.extract_features(X_train, X_test, y_train)
    
    print("Training models...")
    # Train models
    detector.train_models(X_train_combined, y_train)
    
    print("Making predictions...")
    # Make predictions
    y_pred = detector.ensemble_predict(X_test_combined)
    
    # Print results
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    detector.plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main() 
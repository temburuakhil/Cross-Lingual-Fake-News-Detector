from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from fake_news_detector import DravidianFakeNewsDetector
import joblib
import os
import torch
import shap
import json

app = Flask(__name__)

# Load the trained model
model_path = 'models/ensemble_model.joblib'
if os.path.exists(model_path):
    detector = joblib.load(model_path)
else:
    # Train a new model if not exists
    print("Loading data...")
    df = pd.read_csv('balanced_dravidian_fake_news_dataset.csv')
    
    print("Initializing detector...")
    detector = DravidianFakeNewsDetector()
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = detector.preprocess_data(df)
    
    print("Extracting features...")
    X_train_combined, X_test_combined = detector.extract_features(X_train, X_test, y_train)
    
    print("Training models...")
    detector.train_models(X_train_combined, y_train)
    
    # Save the trained model
    os.makedirs('models', exist_ok=True)
    joblib.dump(detector, model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input text and language from the form
        text = request.form['text']
        language = request.form['language']
        
        # Create a DataFrame with the input
        input_data = pd.DataFrame({
            'news_text': [text],
            'language': [language],
            'label': [0]  # Add dummy label for preprocessing
        })
        
        # Combine text and language information
        input_data['combined_text'] = input_data['language'] + ' ' + input_data['news_text']
        
        # Extract features
        input_tfidf = detector.tfidf_vectorizer.transform([input_data['combined_text'].iloc[0]])
        
        def get_bert_embeddings(texts):
            inputs = detector.bert_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                outputs = detector.bert_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).numpy()
        
        input_bert = get_bert_embeddings([input_data['combined_text'].iloc[0]])
        
        # Combine features
        X_test_combined = np.hstack([input_tfidf.toarray(), input_bert])
        
        # Get predictions from all models
        rf_pred = detector.rf_model.predict_proba(X_test_combined)[0]
        xgb_pred = detector.xgb_model.predict_proba(X_test_combined)[0]
        lstm_pred = detector.lstm_model.predict(X_test_combined)[0][0]
        
        # Calculate ensemble prediction and confidence
        weights = [0.4, 0.3, 0.3]  # RF, XGBoost, LSTM weights
        ensemble_pred = (weights[0] * rf_pred[1] + 
                        weights[1] * xgb_pred[1] + 
                        weights[2] * lstm_pred)
        
        # Calculate actual confidence
        confidence = max(ensemble_pred, 1 - ensemble_pred) * 100
        
        # Get the result
        result = "Fake News" if ensemble_pred > 0.5 else "Real News"
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence, 2)
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e),
            'details': error_details
        })

if __name__ == '__main__':
    app.run(debug=True) 
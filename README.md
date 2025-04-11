# Dravidian Fake News Detection System

This project implements a weighted ensemble model for detecting fake news in Dravidian languages (Malayalam, Tamil and Kannada). The system combines multiple models and leverages both traditional NLP features and modern language model embeddings.

## Features

- Multi-model ensemble approach using:
  - Random Forest
  - LSTM (Long Short-Term Memory)
  - XGBoost
- Feature engineering using:
  - TF-IDF features
  - BERT embeddings from IndicBERT-Malayalam model
- Explainable AI using SHAP values
- Support for multiple Dravidian languages
- Comprehensive evaluation metrics and visualizations

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset in CSV format with the following columns:
   - language: The language of the text (Malayalam, Tamil or Kannada)
   - text: The news text
   - label: Binary label (0 for real news, 1 for fake news)

2. Run the main script:
```bash
python fake_news_detector.py
```

3. The script will:
   - Load and preprocess the data
   - Train the ensemble model
   - Generate predictions
   - Create visualizations (confusion matrix and SHAP values)
   - Print classification metrics

## Model Architecture

1. **Feature Extraction**:
   - TF-IDF features (5000 features)
   - BERT embeddings from IndicBERT-Malayalam
   - Combined feature vector

2. **Ensemble Model**:
   - Random Forest (40% weight)
   - XGBoost (30% weight)
   - LSTM (30% weight)

3. **LSTM Architecture**:
   - Embedding layer (5000 vocabulary size)
   - Two LSTM layers with dropout
   - Dense layers for final prediction

## Explainability

The system uses SHAP (SHapley Additive exPlanations) values to explain model predictions, providing insights into which features contribute most to the classification decision.

## Output

The script generates:
1. Classification report with precision, recall, and F1-score
2. Confusion matrix visualization
3. SHAP values plot for feature importance

## Performance

The ensemble approach combines the strengths of different models:
- Random Forest: Good for handling non-linear relationships
- LSTM: Effective for capturing sequential patterns
- XGBoost: Strong performance on structured data

## License

MIT License 
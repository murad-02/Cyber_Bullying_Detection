from flask import Flask, render_template, request, jsonify
import torch
import pickle
import numpy as np
import joblib
import re
import string
from transformers import BertTokenizer
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Global variables for models
label_encoder = None
tokenizer = None
bert_model = None
lstm_model = None
rf_model = None
vectorizer = None
meta_clf = None
vocab = None
device = None

# BERT Classifier class (must match training)
class BERTClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, 5)  # 5 classes
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.linear(cls_output)

# LSTM Classifier class (must match training)
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        pooled = out[:, 0, :]
        dropped = self.dropout(pooled)
        return self.fc(dropped)

def load_models():
    """Load all the trained models and components"""
    global label_encoder, tokenizer, bert_model, lstm_model, rf_model, vectorizer, meta_clf, vocab, device
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load label encoder
        with open("Notebook_File/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained("Notebook_File/saved_tokenizer")
        
        # Load BERT model
        bert_model = BERTClassifier().to(device)
        bert_model.load_state_dict(torch.load("Notebook_File/bert_classifier_state_dict.pth", map_location=device))
        bert_model.eval()
        
        # Try to load LSTM model and vocab (if available)
        try:
            with open("Notebook_File/vocab.pkl", "rb") as f:
                vocab = pickle.load(f)
            lstm_model = LSTMClassifier(vocab_size=len(vocab), num_classes=5).to(device)
            lstm_model.load_state_dict(torch.load("Notebook_File/lstm_model.pth", map_location=device))
            lstm_model.eval()
            print("LSTM model loaded successfully!")
        except FileNotFoundError:
            print("LSTM model files not found. Using BERT only.")
            vocab = {"<PAD>": 0, "<UNK>": 1}
            lstm_model = None
        
        # Try to load Random Forest and TF-IDF vectorizer (if available)
        try:
            rf_model = joblib.load("Notebook_File/random_forest_model.joblib")
            vectorizer = joblib.load("Notebook_File/tfidf_vectorizer.joblib")
            print("Random Forest model loaded successfully!")
        except FileNotFoundError:
            print("Random Forest model files not found. Using BERT only.")
            rf_model = None
            vectorizer = None
        
        # Load ensemble meta-classifier
        meta_clf = joblib.load("Notebook_File/ensemble_meta_clf.joblib")
        
        print("Models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def clean_mixed_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"\@\w+|\#", '', text)
    text = re.sub(r"\d+", '', text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r"[^\u0980-\u09FFa-zA-Z\s]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def encode_sequence(text, vocab, max_len=50):
    """Encode text sequence for LSTM"""
    pad_idx = vocab.get("<PAD>", 0)
    unk = vocab.get("<UNK>", 1)
    tokens = text.split()
    ids = [vocab.get(w, unk) for w in tokens]
    if len(ids) < max_len:
        ids = ids + [pad_idx] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

def predict_cyberbullying(text):
    """Predict cyberbullying using ensemble model"""
    try:
        # Clean text
        cleaned = clean_mixed_text(text)
        
        # BERT prediction
        bert_model.eval()
        with torch.no_grad():
            tokens = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=64)
            inputs = {k: v.to(device) for k, v in tokens.items()}
            bert_logits = bert_model(**inputs)
            bert_probs = torch.softmax(bert_logits, dim=1).cpu().numpy().reshape(1, -1)
        
        # Initialize ensemble components
        ensemble_components = ["BERT"]
        meta_features = [bert_probs]
        
        # LSTM prediction (if available)
        if lstm_model is not None:
            try:
                lstm_model.eval()
                seq_ids = torch.tensor([encode_sequence(cleaned, vocab, max_len=50)], dtype=torch.long).to(device)
                with torch.no_grad():
                    lstm_logits = lstm_model(seq_ids)
                    lstm_probs = torch.softmax(lstm_logits, dim=1).cpu().numpy()
                meta_features.append(lstm_probs)
                ensemble_components.append("LSTM")
            except Exception as e:
                print(f"LSTM prediction failed: {e}")
        
        # Random Forest prediction (if available)
        if rf_model is not None and vectorizer is not None:
            try:
                tfidf_vec = vectorizer.transform([cleaned])
                rf_probs = rf_model.predict_proba(tfidf_vec)
                meta_features.append(rf_probs)
                ensemble_components.append("Random Forest")
            except Exception as e:
                print(f"Random Forest prediction failed: {e}")
        
        # Use ensemble if multiple models are available, otherwise use BERT only
        if len(meta_features) > 1:
            # Combine predictions using meta-classifier
            meta_input = np.hstack(meta_features)
            ensemble_pred_idx = meta_clf.predict(meta_input)[0]
            ensemble_pred = label_encoder.inverse_transform([ensemble_pred_idx])[0]
            ensemble_probs = meta_clf.predict_proba(meta_input)[0]
            
            final_prediction = ensemble_pred
            final_confidence = max(ensemble_probs)
            final_probabilities = ensemble_probs
            model_used = f"Ensemble ({', '.join(ensemble_components)})"
        else:
            # Use BERT only
            bert_pred_idx = np.argmax(bert_probs)
            final_prediction = label_encoder.inverse_transform([bert_pred_idx])[0]
            final_confidence = max(bert_probs.flatten())
            final_probabilities = bert_probs.flatten()
            model_used = "BERT (Ensemble components not available)"
        
        # Create result dictionary with JSON-serializable types
        result = {
            "input_text": text,
            "cleaned_text": cleaned,
            "prediction": str(final_prediction),  # Convert to string
            "confidence": float(final_confidence),  # Convert to Python float
            "all_probabilities": {
                str(label): float(prob) for label, prob in zip(label_encoder.classes_, final_probabilities)
            },
            "model_used": model_used,
            "ensemble_components": ensemble_components
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        result = predict_cyberbullying(text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    # Load models when starting the app
    if load_models():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please check the model files.")

"""
Script to save missing model components from the notebook for the web application.
This script extracts and saves the LSTM model and vocabulary that are needed for the Flask app.
"""

import torch
import pickle
import joblib
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

# Sequence Dataset class
class SeqDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.seqs = [encode_sequence(t, vocab, max_len) for t in texts]
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def build_vocab(texts, max_vocab=8000, min_freq=2):
    """Build vocabulary from training texts"""
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_vocab:
            continue
        if word in vocab:
            continue
        vocab[word] = idx
        idx += 1
    return vocab

def encode_sequence(text, vocab, max_len=50):
    """Encode text sequence for LSTM"""
    pad_idx = vocab["<PAD>"]
    tokens = text.split()
    ids = [vocab.get(w, vocab["<UNK>"]) for w in tokens]
    if len(ids) < max_len:
        ids = ids + [pad_idx] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

def main():
    print("Loading data and models...")
    
    # Load the preprocessed data
    import pandas as pd
    df = pd.read_pickle("Notebook_File/preprocessed_comments.pkl")
    
    # Load label encoder
    with open("Notebook_File/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    # Prepare data
    df = df.copy()
    df['encoded_label'] = label_encoder.transform(df['label'])
    
    X = df['cleaned_text'].tolist()
    y = df['encoded_label'].values
    
    # Split data (same as notebook)
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.17647, stratify=y_temp, random_state=42)
    
    print(f"Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(X_train, max_vocab=8000, min_freq=2)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Save vocabulary
    with open("Notebook_File/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("Vocabulary saved to Notebook_File/vocab.pkl")
    
    # Train LSTM model
    print("Training LSTM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets and dataloaders
    seq_max_len = 50
    batch_size = 64
    
    train_seq_ds = SeqDataset(X_train, y_train, vocab, max_len=seq_max_len)
    train_seq_loader = DataLoader(train_seq_ds, batch_size=batch_size, shuffle=True)
    
    # Initialize LSTM model
    lstm_model = LSTMClassifier(vocab_size=len(vocab), num_classes=len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
    
    # Training function
    def train_epoch(model, loader):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss
    
    # Train for 5 epochs (same as notebook)
    epochs = 5
    for epoch in range(1, epochs+1):
        loss = train_epoch(lstm_model, train_seq_loader)
        print(f"[LSTM] Epoch {epoch} loss={loss:.4f}")
    
    # Save LSTM model
    torch.save(lstm_model.state_dict(), "Notebook_File/lstm_model.pth")
    print("LSTM model saved to Notebook_File/lstm_model.pth")
    
    # Train Random Forest model
    print("Training Random Forest model...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_tfidf, y_train)
    
    # Save Random Forest and TF-IDF vectorizer
    joblib.dump(rf, "Notebook_File/random_forest_model.joblib")
    joblib.dump(vectorizer, "Notebook_File/tfidf_vectorizer.joblib")
    print("Random Forest model saved to Notebook_File/random_forest_model.joblib")
    print("TF-IDF vectorizer saved to Notebook_File/tfidf_vectorizer.joblib")
    
    print("\nAll models saved successfully!")
    print("You can now run the Flask web application with: python app.py")

if __name__ == "__main__":
    main()

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import warnings

# Filter warnings
warnings.filterwarnings('ignore')

# Enable autolog (Ini akan otomatis mencatat params, metrics, dan model)
mlflow.sklearn.autolog()

# Set experiment
# Note: Kita hapus set_tracking_uri agar fleksibel (mengikuti environment variable)
mlflow.set_experiment("sentiment-analysis-playstore")

def load_data():
    """Load preprocessed dataset"""
    print("Loading dataset...")
    # Pastikan file ada di folder yang sama
    df = pd.read_csv('dataset_bersih.csv')
    
    # Bersihkan data NaN
    df = df.dropna(subset=['clean_text', 'sentiment'])
    
    print(f"Dataset shape: {df.shape}")
    return df

def prepare_data(df):
    """Prepare features and labels"""
    print("Preparing data...")
    
    X = df['clean_text']
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    """Train sentiment analysis model"""
    print("Training model...")
    
    with mlflow.start_run(run_name="sentiment-nb-model"):
        # Create pipeline
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB(alpha=1.0))
        ])
        
        # Train
        # Autolog akan bekerja DI SINI saat fungsi .fit() dipanggil
        model.fit(X_train, y_train)
        
        # Predict & Evaluate (Hanya untuk print di layar agar kita tahu hasilnya)
        y_pred = model.predict(X_test)
        
        print(f"\n{'='*50}")
        print(f"Model Training Results (Validation):")
        print(f"{'='*50}")
        print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"F1-Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"{'='*50}\n")
        
        # Get run info untuk konfirmasi
        run = mlflow.active_run()
        print(f"✅ Model logged successfully by Autolog!")
        print(f"📊 Run ID: {run.info.run_id}")
        
        return model

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS MODEL TRAINING (AUTOLOG ONLY)")
    print("="*60 + "\n")
    
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    
    print("\n✅ Training completed!")
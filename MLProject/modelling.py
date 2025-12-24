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

# Set MLflow tracking URI (Pastikan server MLflow sudah jalan di terminal lain: mlflow ui)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("sentiment-analysis-playstore")

# Enable autolog
mlflow.sklearn.autolog()

def load_data():
    """Load preprocessed dataset"""
    print("Loading dataset...")
    # SESUAIKAN PATH FILE DI SINI
    df = pd.read_csv('dataset_bersih.csv')
    
    # Bersihkan data NaN pada kolom penting
    df = df.dropna(subset=['clean_text', 'sentiment'])
    
    print(f"Dataset shape: {df.shape}")
    return df

def prepare_data(df):
    """Prepare features and labels"""
    print("Preparing data...")
    
    # SESUAIKAN NAMA KOLOM DI SINI
    X = df['clean_text']  # Menggunakan text yang sudah dibersihkan
    y = df['sentiment']   # Label sentimen
    
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
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log additional metrics manually
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log parameters manually
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("alpha", 1.0)
        
        print(f"\n{'='*50}")
        print(f"Model Training Results:")
        print(f"{'='*50}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"{'='*50}\n")
        
        # Get run info
        run = mlflow.active_run()
        print(f"✅ Model logged successfully!")
        print(f"📊 Run ID: {run.info.run_id}")
        print(f"🔗 View in MLflow UI: http://127.0.0.1:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        
        # Simpan model secara eksplisit agar foldernya bernama 'model_sentimen'
        # Ini penting agar perintah serving Anda nanti konsisten
        mlflow.sklearn.log_model(model, "model_sentimen")
        
        return model

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS MODEL TRAINING")
    print("="*60 + "\n")
    
    # Load and prepare data
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Train model
    model = train_model(X_train, X_test, y_train, y_test)
    
    print("\n✅ Training completed successfully!")
    print("📁 Check MLflow UI at: http://127.0.0.1:5000")
    print("="*60 + "\n")
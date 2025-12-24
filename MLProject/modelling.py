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
import os

warnings.filterwarnings('ignore')

# Set MLflow tracking untuk CI/CD environment
if os.getenv('GITHUB_ACTIONS'):
    # Running in GitHub Actions
    mlflow.set_tracking_uri("file:./mlruns")
    print("Running in GitHub Actions - using local tracking")
else:
    # Running locally
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print("Running locally - using MLflow server")

mlflow.set_experiment("sentiment-analysis-playstore")

# Enable autolog
mlflow.sklearn.autolog()

def load_data():
    """Load preprocessed dataset"""
    print("Loading dataset...")
    df = pd.read_csv('dataset_preprocessing/dataset_bersih.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def prepare_data(df):
    """Prepare features and labels"""
    print("Preparing data...")
    
    # SESUAIKAN dengan nama kolom di dataset Anda!
    # Contoh jika kolom bernama 'review' dan 'label':
    text_column = 'text'  # GANTI dengan nama kolom text Anda
    label_column = 'sentiment'  # GANTI dengan nama kolom label Anda
    
    X = df[text_column]
    y = df[label_column]
    
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
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log parameters
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
        
        return model

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS MODEL TRAINING")
    print("="*60 + "\n")
    
    try:
        # Load and prepare data
        df = load_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        
        # Train model
        model = train_model(X_train, X_test, y_train, y_test)
        
        print("\n✅ Training completed successfully!")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
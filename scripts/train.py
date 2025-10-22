#!/usr/bin/env python3
import sys
import os

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.models.trainer import FraudModel

def main():
    # Check if data exists
    if not os.path.exists('data/transactions.csv'):
        print("❌ No data found. Running data generator first...")
        from src.data.generator import FraudDataGenerator
        generator = FraudDataGenerator(n_samples=50000)
        df = generator.generate()
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/transactions.csv', index=False)
        print(f"✅ Generated {len(df)} transactions")
    
    # Load data
    print("📊 Loading data...")
    df = pd.read_csv('data/transactions.csv')
    print(f"Loaded {len(df)} transactions with {df['is_fraud'].mean():.2%} fraud rate")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train model
    print("🚀 Training model...")
    model = FraudModel()
    auc_score = model.train(df)
    
    print(f"✅ Training complete!")
    print(f"📈 AUC Score: {auc_score:.4f}")
    print(f"💾 Model saved to models/")

if __name__ == "__main__":
    main()

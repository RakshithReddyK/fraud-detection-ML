## Fraud Detection ML Pipeline
This repository implements an end-to-end fraud detection machine learning pipeline, including:
* Synthetic transaction data generation
* A configurable fraud detection model
* A reproducible training script
* Model evaluation using AUC
* Persisted model artifacts for downstream serving
* The project mirrors how fraud detection systems are structured in real-world ML engineering teams.
## ✨ Overview
This pipeline performs the following steps:
* Checks if ```data/transactions.csv``` exists
* If missing → automatically generates synthetic fraud data using FraudDataGenerator
* Loads the transaction data
* Trains the fraud detection model (FraudModel)
* Computes the AUC score
* Saves the trained model to the ```models/``` directory
* Everything runs through a single command.
## 🧠 Components
## 1. Synthetic Data Generator – ```FraudDataGenerator```
Located in: 
 ```src/data/generator.py```
Responsibilities:
* Creates synthetic transaction-level data
* Default: 50,000 records
* Encodes a realistic fraud distribution via ```is_fraud```
* Saves dataset to ```data/transactions.csv```


The generator makes this project fully self-contained with no external dataset required.
## 2. Model Trainer – ```FraudModel```
Located in:
 ```src/models/trainer.py```
Responsibilities:
* Processes the dataset
* Splits into train/validation
* Trains the ML model
* Computes and returns AUC
* Saves the trained model to ```models/```


Usage:
* model = FraudModel()
* auc_score = model.train(df)
## 3. Training Entrypoint – ```scripts/train.py```
This script orchestrates the entire pipeline:
```!/usr/bin/env python3```
* Automatically generates data if missing
* Loads the dataset
* Trains ```FraudModel```
* Prints AUC
* Saves the model artifact

Run it and everything happens automatically.
## 📂 Project Structure

```
fraud-detection-ML/
├── data/
│   └── transactions.csv        # auto-generated if missing
├── models/
│   └── fraud_model.pkl         # saved trained model
├── src/
│   ├── data/
│   │   └── generator.py        # FraudDataGenerator
│   ├── models/
│   │   └── trainer.py          # FraudModel
│   └── __init__.py
├── scripts/
│   └── train.py                # full pipeline entrypoint
├── requirements.txt
└── README.md
```


## ⚙️ Setup Instructions
1. Clone the repository
```bash
git clone https://github.com/<your-username>/fraud-detection-ML.git
cd fraud-detection-ML
```
2. Create virtual environment
```bash python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```
3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
4. Run the training pipeline
```bash
scripts/train.py
```
🧪 Example Output
```❌ No data found. Running data generator first...
✅ Generated 50000 transactions
📊 Loading data...
Loaded 50000 transactions with 2.15% fraud rate
🚀 Training model...
✅ Training complete!
📈 AUC Score: 0.9473
💾 Model saved to models/
```
(AUC and fraud rate will vary based on generator settings.)
## 🎯 Why This Project Matters
This project showcases real-world ML engineering practices:
Data generation + model training flow
Modular architecture
Single-entrypoint automation
Proper metric reporting (AUC)
Model artifact management
Production-style folder layout
## 🚀 Future Enhancements
You can extend this project with:
* FastAPI model-serving API
* MLflow model tracking
* Real-time fraud scoring with Kafka/Kinesis
* Feature engineering module
* Hyperparameter optimization

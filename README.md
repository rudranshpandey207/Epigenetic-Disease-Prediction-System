# 🧬 Epigenetic Disease Prediction System

### 🧠 Overview

This project predicts disease risk (Alzheimer's Disease and Prostate Cancer) using **epigenetic methylation data** with a **privacy-preserving encryption workflow**. Users can encrypt their sensitive health data locally using **Fernet symmetric encryption** before uploading. It combines **Machine Learning**, **Bioinformatics**, **Data Privacy**, and a **Streamlit Web App**.

---

## 🚀 Features

- **Data Source:** Epigenetic datasets from GEO (GSE80970 for Alzheimer's, GSE26126 for Prostate Cancer)
- **Model Training:** Random Forest Classifiers (2000 features for Prostate, 500 for Alzheimer's)
- **Privacy-Preserving:** Decrypt-Predict-Encrypt workflow for sensitive data
- **Encryption:** Fernet symmetric encryption with PBKDF2 key derivation
- **Dual Mode:** Standard CSV upload or encrypted file upload
- **Web Application:** Streamlit app for real-time disease prediction
- **User Input:** Upload epigenetic methylation data (CSV format)
- **Disease Selection:** Choose between Alzheimer's and Prostate Cancer for prediction
- **Local Encryption/Decryption:** Tools provided for client-side data security

---

## 🧱 Project Architecture

```
User (Local Machine)
   │
   ├── Encrypt data locally (optional, for privacy)
   ├── Upload to Streamlit app (CSV or .encrypted)
   ▼
Streamlit Server
   ├── Standard Mode: Process CSV directly
   ├── Encrypted Mode: Decrypt in memory → Predict → Re-encrypt
   ├── Return results (plain or encrypted)
   ▼
User (Local Machine)
   └── Decrypt results locally (if encrypted mode)

ML Models:
   ├── Alzheimer's: Random Forest (500 features, ~92% accuracy)
   └── Prostate: Random Forest (2000 features, ~94% accuracy)
```

---

## ⚙️ Technologies Used

| Component      | Technology                                                   |
| -------------- | ------------------------------------------------------------ |
| Data Source    | GEOparse, GSE80970 (Alzheimer's), GSE26126 (Prostate Cancer) |
| ML Framework   | scikit-learn (Random Forest)                                 |
| Security       | Fernet (AES-128-CBC + HMAC-SHA256)                           |
| Key Derivation | PBKDF2-HMAC-SHA256 (100,000 iterations)                      |
| Web Framework  | Streamlit                                                    |
| Environment    | Local deployment (Python 3.8+)                               |

---

## 🧩 Folder Structure

```
Epigenetic_Disease_Predictor/
│
├── src/
│   ├── app.py                     # Streamlit web app
│   ├── encryption_utils.py        # Fernet encryption utilities
│   ├── encrypt_data_local.py      # Client-side encryption tool
│   └── decrypt_result_local.py    # Client-side decryption tool
├── models/
│   ├── alzheimer_model.joblib     # Alzheimer's RF model
│   ├── alzheimer_scaler.joblib    # Feature scaler
│   ├── prostate_rf_model_2000f_70_30.joblib  # Prostate RF model
│   └── prostate_rf_scaler_2000f_70_30.joblib # Feature scaler
├── test/                          # Test datasets
├── ENCRYPTION_GUIDE.md            # Detailed encryption docs
├── QUICKSTART.md                  # Quick start guide
├── README.md                      # Project documentation
└── requirements.txt               # Dependencies
```

---

## 💻 Setup Instructions

### **1️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2️⃣ Run the Streamlit Web App**

```bash
streamlit run app.py
```

Access the app at **[http://localhost:8501](http://localhost:8501)**

---

## 🌐 Web App Usage

### Standard Mode (Quick Testing)

1. Launch the app (`streamlit run src/app.py`)
2. Select "Standard Upload" in sidebar
3. Choose disease model (Alzheimer's / Prostate Cancer)
4. Upload CSV file
5. Click **🚀 Predict Disease Status**
6. View results immediately

### Encrypted Mode (Privacy-Preserving)

1. **Encrypt locally:** `python src/encrypt_data_local.py your_data.csv`
2. Save password and salt
3. Launch app and select "🔒 Encrypted Upload"
4. Upload `.encrypted` file
5. Enter password and salt
6. Click **🚀 Predict**
7. Copy encrypted result
8. **Decrypt locally:** `python src/decrypt_result_local.py`
9. View decrypted results with accuracy metrics

---

## 🔒 Security Features

- **Fernet Encryption:** AES-128-CBC with HMAC-SHA256 authentication
- **Password-Based:** PBKDF2 key derivation (100,000 iterations)
- **In-Memory Decryption:** Server never stores unencrypted data
- **End-to-End Privacy:** Data encrypted on client, decrypted in RAM, results re-encrypted
- **Local Control:** Only you can decrypt your results with your password
- **Zero Knowledge:** Server operators cannot access your data

---

## 📊 Results

| Disease         | Model Used    | Features | Training Acc | Test Acc |
| --------------- | ------------- | -------- | ------------ | -------- |
| Alzheimer's     | Random Forest | 500      | ~85%         | ~82%     |
| Prostate Cancer | Random Forest | 2000     | ~98%         | ~94%     |

---

## 📜 Future Improvements

- Add support for more diseases (Parkinson’s, Diabetes)
- Integrate a blockchain layer for model integrity verification
- Add SHAP/Feature Importance visualization in Streamlit
- Deploy Streamlit app on Streamlit Cloud or AWS

---

## 👥 Authors

- **Rudransh Pandey** – Machine Learning & Backend
- **Ishan Mittal** – Data preprocessing, encryption
- **Himangi Rawat** - Web app integration

---

## 🧾 License

This project is for academic and research purposes only.
All datasets are publicly available from the NCBI GEO repository.

---


# 🧬 Epigenetic Disease Prediction System

### 🧠 Overview

This project predicts disease risk (Alzheimer's Disease and Prostate Cancer) using **epigenetic methylation data** and ensures **data security** through RSA encryption and cloud storage integration. It combines **Machine Learning**, **Bioinformatics**, **Data Security**, and a **Streamlit Web App**.

---

## 🚀 Features

* **Data Source:** Epigenetic datasets from GEO (GSE80970 for Alzheimer's, GSE26126 for Prostate Cancer)
* **Model Training:** Random Forest Classifiers trained in Kaggle (due to high RAM requirements)
* **Encryption:** RSA encryption of trained models and datasets
* **Cloud Storage:** Encrypted models uploaded to Google Drive
* **Web Application:** Streamlit app built and run locally in VS Code for real-time disease prediction
* **User Input:** Upload epigenetic methylation data (CSV format)
* **Disease Selection:** Choose between Alzheimer's and Prostate Cancer for prediction

---

## 🧱 Project Architecture

```
Kaggle (Cloud ML)
   │
   ├── Fetch epigenetic datasets (GEO)
   ├── Preprocess and train ML models
   ├── Encrypt models & datasets (RSA)
   ├── Upload encrypted files to Google Drive
   ▼
VS Code (Local Deployment)
   ├── Download & decrypt encrypted models
   ├── Launch Streamlit web app
   ├── User uploads epigenetic data CSV
   └── Predict disease status (Control / Diseased)
```

---

## ⚙️ Technologies Used

| Component     | Technology                                                   |
| ------------- | ------------------------------------------------------------ |
| Data Source   | GEOparse, GSE80970 (Alzheimer's), GSE26126 (Prostate Cancer) |
| ML Framework  | scikit-learn (Random Forest, XGBoost optional)               |
| Security      | RSA (PyCryptodome library)                                   |
| Cloud         | Google Drive API (PyDrive2)                                  |
| Web Framework | Streamlit                                                    |
| Environment   | Kaggle (training), VS Code (app deployment)                  |

---

## 🧩 Folder Structure

```
Epigenetic_Disease_Predictor/
│
├── app.py                         # Streamlit web app
├── alzheimers_model.pkl           # Decrypted Alzheimer’s model
├── prostate_model.pkl             # Decrypted Prostate model
├── private_key.pem / public_key.pem  # RSA keys
├── encrypted_model.bin            # Encrypted model files (from Kaggle)
├── datasets/                      # Epigenetic datasets (CSV)
├── README.md                      # Project documentation
└── requirements.txt               # Dependencies
```

---

## 💻 Setup Instructions

### **1️⃣ Train Models on Kaggle**

* Open Kaggle Notebook
* Run `train_and_encrypt.ipynb` to:

  * Fetch GEO data
  * Train ML models
  * Encrypt models & datasets
  * Upload to Google Drive

### **2️⃣ Download Encrypted Files**

From Kaggle/Google Drive, download:

* `encrypted_alzheimers_model.bin`
* `encrypted_prostate_model.bin`
* `private_key.pem`
* `public_key.pem`

### **3️⃣ Decrypt Locally in VS Code**

```bash
python decrypt_models.py
```

This will create `alzheimers_model.pkl` and `prostate_model.pkl` for app use.

### **4️⃣ Run the Streamlit Web App**

```bash
streamlit run app.py
```

Access the app at **[http://localhost:8501](http://localhost:8501)**

---

## 🌐 Web App Usage

1. Launch the app (`streamlit run app.py`)
2. Choose a disease model (Alzheimer’s / Prostate Cancer)
3. Upload your epigenetic data (CSV)
4. Click **Predict**
5. View prediction results (Control / Disease)

---

## 🔒 Security Features

* **RSA Encryption:** Protects both model and dataset files.
* **Cloud Integration:** Encrypted files stored on Google Drive for secure access.
* **Decryption Key Control:** Only the private key holder can decrypt and use the models.

---

## 📊 Results

| Disease         | Model Used    | Accuracy |
| --------------- | ------------- | -------- |
| Alzheimer's     | Random Forest | ~90%     |
| Prostate Cancer | Random Forest | ~88%     |

---

## 📜 Future Improvements

* Add support for more diseases (Parkinson’s, Diabetes)
* Integrate a blockchain layer for model integrity verification
* Add SHAP/Feature Importance visualization in Streamlit
* Deploy Streamlit app on Streamlit Cloud or AWS

---

## 👥 Authors

* **Rudransh Pandey** – Machine Learning & Backend
* **Team Members (if any)** – Data preprocessing, encryption, and web app integration

---

## 🧾 License

This project is for academic and research purposes only.
All datasets are publicly available from the NCBI GEO repository.

---

## 💡 Citation

> Pandey, R. (2025). *Epigenetic-Based Disease Prediction Using Machine Learning and Secure Cloud Storage.* Minor Project Report, [Your College Name].

# 🚀 Quick Start Guide - Privacy-Preserving Epigenetic Prediction

## ⚡ 5-Minute Setup

### Prerequisites

- Python 3.8+
- Anaconda/Miniconda (optional but recommended)
- Git (optional)

---

## 📦 Installation

### Step 1: Clone/Download Project

```bash
cd F:\Hackathon\Minor
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues, install packages individually:

```bash
pip install streamlit pandas scikit-learn numpy pytesseract pillow PyPDF2 python-docx cryptography joblib
```

### Step 3: Install Tesseract (for Image OCR)

**Windows:**

```bash
choco install tesseract
```

Or download from: https://github.com/UB-Mannheim/tesseract/wiki

**Verify installation:**

```bash
tesseract --version
```

---

## 🎯 Usage Modes

### Mode 1: Standard Upload (Quick Testing)

**For:** Non-sensitive or anonymized data

```bash
# Start the app
streamlit run src/app.py
```

1. Select **"Standard Upload"** in sidebar
2. Choose disease model (Alzheimer's or Prostate Cancer)
3. Upload CSV/PDF/Word/Image file
4. Click **"🚀 Predict Disease Status"**
5. View results immediately

**Test with:**

```bash
# Use test files
test/alz dataset.csv
test/TCGA-PRAD_processed_for_cloud.csv
```

---

### Mode 2: Encrypted Upload (Maximum Privacy)

**For:** Sensitive health data requiring privacy

#### Step 1: Encrypt Your Data

```bash
cd src
python encrypt_data_local.py ../test/your_data.csv
```

**Output:**

```
✅ Encryption successful!
Password: [YOUR_PASSWORD]
Salt: [YOUR_SALT]
Encrypted file: your_data.csv.encrypted
```

**⚠️ IMPORTANT:** Save password and salt securely!

#### Step 2: Upload to App

```bash
streamlit run app.py
```

1. Select **"🔒 Encrypted Upload"** in sidebar
2. Upload `.encrypted` file
3. Enter password and salt
4. Click **"🚀 Predict"**
5. Copy encrypted result

#### Step 3: Decrypt Results Locally

```bash
python decrypt_result_local.py
```

1. Paste encrypted result
2. Enter password
3. Enter salt
4. View decrypted predictions

---

## 📊 Supported File Formats

| Format | Extension               | Notes                       |
| ------ | ----------------------- | --------------------------- |
| CSV    | `.csv`                  | Direct import, recommended  |
| PDF    | `.pdf`                  | Text extraction from tables |
| Word   | `.docx`                 | Table and text extraction   |
| Images | `.png`, `.jpg`, `.tiff` | OCR with Tesseract          |

---

## 🧪 Quick Test

### Test Standard Mode

```bash
# Start app
streamlit run src/app.py

# Upload test file (in app)
test/alz dataset.csv

# Select: "Alzheimer's Disease"
# Click: "🚀 Predict"
```

### Test Encryption Workflow

```bash
# 1. Generate test data
cd test
python gen_balanced.py

# 2. Encrypt
cd ../src
python encrypt_data_local.py ../test/sample_prostate_data.csv
# Password: TestPassword123
# Save the salt shown!

# 3. Upload in app
streamlit run app.py
# Mode: Encrypted Upload
# Upload: sample_prostate_data.csv.encrypted
# Enter password and salt

# 4. Decrypt result
python decrypt_result_local.py
# Paste encrypted result
# Enter password and salt
```

---

## 🎨 App Features

### Main Interface

- **Disease Selection:** Alzheimer's or Prostate Cancer
- **Privacy Mode:** Standard or Encrypted upload
- **Multi-format:** CSV, PDF, Word, Images
- **Real-time:** Instant predictions

### Results Display

- **Summary Metrics:** Healthy vs Disease counts
- **Individual Results:** Per-sample predictions
- **Confidence Scores:** Prediction reliability
- **Feature Importance:** Top CpG sites
- **Export Options:** CSV download

---

## 🔍 Troubleshooting

### App Won't Start

```bash
# Check Streamlit installation
streamlit --version

# Reinstall if needed
pip install --upgrade streamlit
```

### "Model not found" Error

**Solution:** Ensure model files exist in `models/` directory:

- `prostate_rf_model_2000f_70_30.joblib`
- `prostate_rf_scaler_2000f_70_30.joblib`
- `alzheimer_model.joblib`
- `alzheimer_scaler.joblib`

### "Tesseract not found" Error

**Solution:** Update path in `app.py`:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### "Feature mismatch" Warning

**Solution:**

1. Verify correct disease model selected
2. Check CSV format (CpG site names)
3. Try other test files

### Decryption Failed

**Solution:**

1. Verify password is correct
2. Verify salt is copied correctly (no spaces)
3. Check encrypted result is complete

---

## 📋 Data Format Requirements

### CSV Structure

```csv
SampleID,cg00000029,cg00000165,cg00000236,...
Patient001,0.523,0.678,0.234,...
Patient002,0.445,0.789,0.567,...
```

**Requirements:**

- First column: Sample identifiers
- Remaining columns: CpG methylation values (0-1)
- Prostate model: 2000 features
- Alzheimer's model: 500 features

### Missing Features

Don't worry! The app automatically:

- Fills missing features with zeros
- Ignores extra features
- Shows feature match percentage

---

## 🔐 Security Best Practices

### For Encrypted Mode:

1. **Strong Passwords:** Use 12+ characters with mixed case, numbers, symbols
2. **Salt Storage:** Store salt separately from encrypted file
3. **Secure Transmission:** Use HTTPS for file uploads
4. **Local Decryption:** Always decrypt results locally
5. **Clean Up:** Delete encrypted files after use

### Password Examples:

- ✅ Good: `Epigen#2025!Secure`
- ❌ Bad: `password123`

---

## 💡 Pro Tips

### Speed Up Predictions

- Use CSV format (fastest)
- Reduce image resolution before OCR
- Process multiple samples in one file

### Better Accuracy

- Use datasets from same source as training data
- Ensure proper CpG site nomenclature
- Verify methylation values are 0-1 range

### Privacy Enhancement

- Use encrypted mode for all sensitive data
- Generate new password for each dataset
- Use offline machine for decryption

---

## 📞 Getting Help

### Common Issues

1. Check `ENCRYPTION_GUIDE.md` for detailed docs
2. Review `README.md` for project overview
3. Check console output for error messages

### Error Messages

- Import errors → Reinstall requirements
- Model errors → Check model files exist
- Format errors → Verify CSV structure
- Encryption errors → Check password/salt

---

## 🎓 Learning Resources

### Understanding Epigenetics

- DNA Methylation basics
- CpG sites and disease association
- Beta value interpretation

### Understanding Encryption

- Fernet symmetric encryption
- PBKDF2 key derivation
- Decrypt-Predict-Encrypt workflow

---

## 📈 Next Steps

After getting started:

1. **Test with your data:** Prepare CSV with your methylation values
2. **Explore results:** Understand confidence scores and feature importance
3. **Secure workflow:** Practice encryption/decryption process
4. **Customize:** Modify models or add new disease types

---

## ✅ Checklist

Before first use:

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Tesseract OCR installed (if using images)
- [ ] Model files present in `models/` directory
- [ ] Test files available in `test/` directory
- [ ] Understand CSV format requirements
- [ ] Practice encryption workflow with test data

---

## 🚀 Launch Commands

**Standard Mode:**

```bash
streamlit run src/app.py
```

**Custom Port:**

```bash
streamlit run src/app.py --server.port 8080
```

**Network Access:**

```bash
streamlit run src/app.py --server.address 0.0.0.0
```

**Debug Mode:**

```bash
streamlit run src/app.py --logger.level=debug
```

---

**Happy Predicting! 🧬**

For detailed documentation, see `ENCRYPTION_GUIDE.md`

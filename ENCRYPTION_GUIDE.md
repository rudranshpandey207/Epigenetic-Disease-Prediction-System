# 🔒 Privacy-Preserving Encryption Workflow

## Overview

This project implements a **Decrypt-Predict-Encrypt** workflow for privacy-preserving epigenetic disease predictions. Your sensitive health data is encrypted on your local machine, processed securely on the server, and results are re-encrypted before being returned to you.

## 🔐 How It Works

```
┌──────────────────┐
│  Your Computer   │
│  1. Encrypt Data │
└────────┬─────────┘
         │ Upload .encrypted file + password
         ▼
┌──────────────────┐
│   Server Side    │
│  2. Decrypt      │
│  3. Predict      │
│  4. Re-Encrypt   │
└────────┬─────────┘
         │ Return encrypted result
         ▼
┌──────────────────┐
│  Your Computer   │
│  5. Decrypt      │
│  6. View Results │
└──────────────────┘
```

### Key Privacy Features

- ✅ **Local Encryption**: Data encrypted on your device before upload
- ✅ **In-Memory Decryption**: Server decrypts in RAM (never saved to disk)
- ✅ **Zero Storage**: No unencrypted data stored on server
- ✅ **Re-Encryption**: Results encrypted with your key before transmission
- ✅ **Local Decryption**: Only you can decrypt the results

## 📋 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Your CSV file should contain:

- First column: Sample IDs
- Remaining columns: CpG site methylation values (0-1)
- Format: `cg00000029, cg00000165, ...` (2000 features for Prostate, 500 for Alzheimer's)

Example:

```csv
SampleID,cg00000029,cg00000165,cg00000236,...
Sample1,0.523,0.678,0.234,...
Sample2,0.445,0.789,0.567,...
```

## 🔐 Privacy-Preserving Workflow

### Step 1: Encrypt Your Data Locally

```bash
cd src
python encrypt_data_local.py ../test/your_data.csv
```

**What happens:**

1. Script prompts for password (minimum 8 characters)
2. Derives encryption key from password using PBKDF2
3. Encrypts CSV file with Fernet symmetric encryption
4. Creates `your_data.csv.encrypted` file
5. Displays password and salt (SAVE THESE!)

**Example output:**

```
✅ Encryption successful!

📦 ENCRYPTED FILE DETAILS
Encrypted file: your_data.csv.encrypted
File size: 54321 bytes

🔑 IMPORTANT - Save these securely:
Password: YourSecurePassword123
Salt: dGhpc2lzYXNhbHQ=
```

### Step 2: Upload to Streamlit App

1. Launch the app:

```bash
streamlit run src/app.py
```

2. In the sidebar, select **"🔒 Encrypted Upload (Privacy-Preserving)"**

3. Upload your `.encrypted` file

4. Enter your **password** and **salt** in the secure fields

5. The app will:
   - Decrypt file in memory (RAM only)
   - Run prediction
   - Clear decrypted data from memory
   - Re-encrypt results with your key

### Step 3: Get Encrypted Results

After prediction, you'll see an encrypted result string like:

```
gAAAAABhx7y8ZX...encrypted_data...3mK9pQ==
```

**Copy or download this encrypted result.**

### Step 4: Decrypt Results Locally

```bash
cd src
python decrypt_result_local.py
```

**Enter when prompted:**

1. The encrypted result string
2. Your password
3. Your salt value

**Example output:**

```
🔬 DECRYPTED PREDICTION RESULTS

Disease Model: Prostate Cancer
Total Samples: 10

📊 PREDICTION SUMMARY
✅ Healthy (Control): 6 samples
🔴 Disease Positive: 4 samples

Disease Rate: 40.0%

📋 DETAILED PREDICTIONS
Sample 1: ✅ Healthy
Sample 2: 🔴 Disease
Sample 3: ✅ Healthy
...
```

## 🛠️ Command-Line Tools

### `encrypt_data_local.py`

**Purpose:** Encrypt CSV files before uploading

**Usage:**

```bash
# Interactive mode
python encrypt_data_local.py

# Command-line mode
python encrypt_data_local.py path/to/data.csv
```

**Features:**

- Password-based encryption (PBKDF2 + Fernet)
- Salt generation for key derivation
- Progress indicators
- Validation checks

---

### `decrypt_result_local.py`

**Purpose:** Decrypt prediction results locally

**Usage:**

```bash
# Interactive mode
python decrypt_result_local.py

# With encrypted result
python decrypt_result_local.py "gAAAAABhx7y8..."
```

**Features:**

- Secure password input
- Result formatting
- Export to JSON
- Error handling

---

## 🔒 Security Details

### Encryption Algorithm

- **Method:** Fernet (Symmetric Encryption)
- **Key Derivation:** PBKDF2-HMAC-SHA256
- **Iterations:** 100,000
- **Salt Length:** 16 bytes
- **Key Length:** 32 bytes

### Security Properties

1. **Confidentiality:** AES-128-CBC encryption
2. **Authenticity:** HMAC-SHA256 signature
3. **Freshness:** Timestamp validation
4. **Password Security:** PBKDF2 with 100k iterations

### Threat Model Protection

✅ **Server Compromise:** Server never sees unencrypted data at rest  
✅ **Network Sniffing:** Data encrypted in transit  
✅ **Data Breaches:** Encrypted files useless without password  
✅ **Insider Threats:** Server operators cannot access your data

⚠️ **Not Protected Against:**

- Compromised client machine
- Keyloggers capturing password
- Side-channel attacks during processing

---

## 📊 Standard Upload Mode

For non-sensitive or already anonymized data, use standard mode:

1. Select **"Standard Upload"** in sidebar
2. Upload CSV files directly
3. View results immediately in browser

**Supported formats:**

- 📄 CSV (direct import)
---

## 🧪 Testing

### Generate Test Files

```bash
cd test
python generate_test_files.py
```

Creates:

- `test_data.pdf` - PDF with methylation data
- `test_data.docx` - Word document
- `test_data.png` - Image for OCR

### Test Encryption Workflow

```bash
# 1. Create test CSV
cd test
python gen_balanced.py

# 2. Encrypt it
cd ../src
python encrypt_data_local.py ../test/sample_data.csv

# 3. Upload to app (manual step)
streamlit run app.py

# 4. Decrypt result
python decrypt_result_local.py
```

---

## ⚙️ Configuration

### Encryption Settings

Edit `encryption_utils.py` to modify:

```python
# Increase iterations (slower but more secure)
iterations=100000  # Default

# Change salt length
salt = os.urandom(16)  # 16 bytes default
```

### App Settings

Edit `app.py`:

```python
# Enable/disable encryption mode by default
privacy_mode = "🔒 Encrypted Upload"  # or "Standard Upload"
```

---

## 🐛 Troubleshooting

### "Decryption failed: Invalid token"

**Cause:** Wrong password or salt  
**Solution:** Double-check password and salt values

### "Feature mismatch" warning

**Cause:** CSV has wrong CpG features  
**Solution:**

- Switch disease model (Alzheimer's ↔ Prostate)
- Verify dataset preprocessing

### OCR errors with images

**Cause:** Tesseract not installed or wrong path  
**Solution:**

```bash
# Windows
choco install tesseract

# Verify path in app.py:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### "ModuleNotFoundError: cryptography"

**Solution:**

```bash
pip install cryptography
```

---

## 📖 API Reference

### `encryption_utils.py`

#### `derive_key_from_password(password, salt=None)`

Derives encryption key from password using PBKDF2.

**Parameters:**

- `password` (str): User password
- `salt` (bytes, optional): Salt for key derivation

**Returns:** `(key, salt)` tuple

---

#### `encrypt_file(file_path, key, output_path=None)`

Encrypts a file using Fernet.

**Parameters:**

- `file_path` (str): Path to file to encrypt
- `key` (bytes): Encryption key
- `output_path` (str, optional): Output file path

**Returns:** Path to encrypted file

---

#### `decrypt_data(encrypted_data, key)`

Decrypts data in memory.

**Parameters:**

- `encrypted_data` (bytes): Encrypted data
- `key` (bytes): Decryption key

**Returns:** Decrypted bytes

---

#### `encrypt_result(result, key)`

Encrypts prediction result dictionary.

**Parameters:**

- `result` (dict): Prediction results
- `key` (bytes): Encryption key

**Returns:** Base64-encoded encrypted string

---

#### `decrypt_result(encrypted_result, key)`

Decrypts prediction results.

**Parameters:**

- `encrypted_result` (str): Encrypted result string
- `key` (bytes): Decryption key

**Returns:** Result dictionary

---

## 📚 Additional Resources

- [Fernet Specification](https://github.com/fernet/spec/)
- [PBKDF2 RFC](https://tools.ietf.org/html/rfc2898)
- [OWASP Cryptographic Storage](https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html)

---

## 🤝 Contributors

- Rudransh Pandey
- Ishan Mittal
- Himangi Rawat

---

## ⚠️ Disclaimer

This tool is for **research purposes only**. Not intended for clinical diagnosis. Consult healthcare professionals for medical advice.

---

## 📄 License

[Add your license here]

---

**Last Updated:** 2025-01-27

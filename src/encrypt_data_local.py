"""
Local encryption tool for users to encrypt their data files before uploading
Run this script to encrypt your CSV file with a password
"""
import sys
import os
from pathlib import Path
from encryption_utils import derive_key_from_password, encrypt_file
import base64


def encrypt_csv_for_upload(csv_path: str, password: str, output_path: str = None):
    """
    Encrypt a CSV file with a password for secure upload
    
    Args:
        csv_path: Path to the CSV file to encrypt
        password: Password to use for encryption
        output_path: Optional output path for encrypted file
    
    Returns:
        Tuple of (encrypted_file_path, salt_base64)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    # Derive encryption key from password
    key, salt = derive_key_from_password(password)
    
    # Set output path
    if output_path is None:
        output_path = csv_path + '.encrypted'
    
    # Encrypt the file
    encrypted_path = encrypt_file(csv_path, key, output_path)
    
    # Encode salt as base64 for easy transport
    salt_b64 = base64.b64encode(salt).decode()
    
    return encrypted_path, salt_b64


def main():
    print("=" * 60)
    print("🔐 PRIVACY-PRESERVING DATA ENCRYPTION TOOL")
    print("=" * 60)
    print()
    
    if len(sys.argv) < 2:
        print("Usage: python encrypt_data_local.py <csv_file>")
        print()
        print("Example: python encrypt_data_local.py my_data.csv")
        print()
        csv_file = input("Enter the path to your CSV file: ").strip()
    else:
        csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found!")
        sys.exit(1)
    
    print(f"📁 File to encrypt: {csv_file}")
    print()
    
    # Get password from user
    import getpass
    password = getpass.getpass("🔑 Enter encryption password: ")
    password_confirm = getpass.getpass("🔑 Confirm password: ")
    
    if password != password_confirm:
        print("❌ Passwords do not match!")
        sys.exit(1)
    
    if len(password) < 8:
        print("❌ Password must be at least 8 characters long!")
        sys.exit(1)
    
    print()
    print("⏳ Encrypting your data...")
    
    try:
        encrypted_path, salt = encrypt_csv_for_upload(csv_file, password)
        
        print()
        print("✅ Encryption successful!")
        print()
        print("=" * 60)
        print("📦 ENCRYPTED FILE DETAILS")
        print("=" * 60)
        print(f"Encrypted file: {encrypted_path}")
        print(f"File size: {os.path.getsize(encrypted_path)} bytes")
        print()
        print("🔑 IMPORTANT - Save these securely:")
        print("-" * 60)
        print(f"Password: {password}")
        print(f"Salt: {salt}")
        print("-" * 60)
        print()
        print("📤 NEXT STEPS:")
        print("1. Upload the encrypted file (.encrypted) to the Streamlit app")
        print("2. Enter your password in the secure field")
        print("3. Enter the salt value shown above")
        print("4. The app will decrypt, predict, and re-encrypt your results")
        print()
        print("⚠️  SECURITY NOTES:")
        print("- Keep your password and salt safe")
        print("- The encrypted file cannot be read without the password")
        print("- The server never stores your decryption key")
        print("- Results are encrypted before sending back to you")
        print()
        
    except Exception as e:
        print(f"❌ Error during encryption: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

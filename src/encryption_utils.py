"""
Encryption utilities for privacy-preserving predictions
Uses Fernet (symmetric encryption) for file and result encryption
"""
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import os


def generate_encryption_key():
    """Generate a new Fernet encryption key"""
    return Fernet.generate_key()


def derive_key_from_password(password: str, salt: bytes = None) -> tuple:
    """
    Derive a Fernet key from a password using PBKDF2
    Returns: (key, salt)
    """
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key, salt


def encrypt_data(data: bytes, key: bytes) -> bytes:
    """Encrypt data using Fernet symmetric encryption"""
    f = Fernet(key)
    encrypted_data = f.encrypt(data)
    return encrypted_data


def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
    """Decrypt data using Fernet symmetric encryption"""
    f = Fernet(key)
    decrypted_data = f.decrypt(encrypted_data)
    return decrypted_data


def encrypt_file(file_path: str, key: bytes, output_path: str = None):
    """Encrypt a file and save to output path"""
    if output_path is None:
        output_path = file_path + '.encrypted'
    
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    encrypted_data = encrypt_data(file_data, key)
    
    with open(output_path, 'wb') as f:
        f.write(encrypted_data)
    
    return output_path


def decrypt_file(encrypted_file_path: str, key: bytes, output_path: str = None):
    """Decrypt a file and save to output path"""
    if output_path is None:
        output_path = encrypted_file_path.replace('.encrypted', '.decrypted')
    
    with open(encrypted_file_path, 'rb') as f:
        encrypted_data = f.read()
    
    decrypted_data = decrypt_data(encrypted_data, key)
    
    with open(output_path, 'wb') as f:
        f.write(decrypted_data)
    
    return output_path


def encrypt_text(text: str, key: bytes) -> str:
    """Encrypt text and return base64 encoded string"""
    encrypted = encrypt_data(text.encode(), key)
    return base64.urlsafe_b64encode(encrypted).decode()


def decrypt_text(encrypted_text: str, key: bytes) -> str:
    """Decrypt base64 encoded encrypted text"""
    encrypted_bytes = base64.urlsafe_b64decode(encrypted_text.encode())
    decrypted = decrypt_data(encrypted_bytes, key)
    return decrypted.decode()


def encrypt_result(result: dict, key: bytes) -> str:
    """Encrypt prediction result dictionary"""
    import json
    result_json = json.dumps(result)
    return encrypt_text(result_json, key)


def decrypt_result(encrypted_result: str, key: bytes) -> dict:
    """Decrypt prediction result dictionary"""
    import json
    decrypted_json = decrypt_text(encrypted_result, key)
    return json.loads(decrypted_json)

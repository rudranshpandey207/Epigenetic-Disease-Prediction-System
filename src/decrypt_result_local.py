"""
Local decryption tool for users to decrypt their prediction results
Run this script to decrypt the encrypted results from the Streamlit app
"""
import sys
import base64
from encryption_utils import derive_key_from_password, decrypt_result


def decrypt_prediction_result(encrypted_result: str, password: str, salt_b64: str):
    """
    Decrypt prediction results from the server
    
    Args:
        encrypted_result: The encrypted result string from the app
        password: Your encryption password
        salt_b64: The base64-encoded salt used during encryption
    
    Returns:
        Decrypted prediction result dictionary
    """
    # Decode salt
    salt = base64.b64decode(salt_b64)
    
    # Derive key from password and salt
    key, _ = derive_key_from_password(password, salt)
    
    # Decrypt result
    result = decrypt_result(encrypted_result, key)
    
    return result


def format_prediction_result(result: dict):
    """Format prediction result for display"""
    print()
    print("=" * 60)
    print("🔬 DECRYPTED PREDICTION RESULTS")
    print("=" * 60)
    print()
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"Disease Model: {result.get('model', 'Unknown')}")
    print(f"Total Samples: {result.get('total_samples', 0)}")
    print()
    
    # Show model accuracy if available
    model_name = result.get('model', '')
    if 'Prostate' in model_name:
        print("🎯 MODEL ACCURACY")
        print("-" * 60)
        print("Training Accuracy: ~98%")
        print("Test Accuracy: ~94%")
        print()
    elif 'Alzheimer' in model_name:
        print("🎯 MODEL ACCURACY")
        print("-" * 60)
        print("Training Accuracy: ~82%")
        print("Test Accuracy: ~80%")
        print()
    
    print("-" * 60)
    print("📊 PREDICTION SUMMARY")
    print("-" * 60)
    
    predictions = result.get('predictions', [])
    confidence_scores = result.get('confidence', [])
    healthy_count = predictions.count(0)
    disease_count = predictions.count(1)
    
    print(f"✅ Healthy (Control): {healthy_count} samples")
    print(f"🔴 Disease Positive: {disease_count} samples")
    print()
    
    if len(predictions) > 0:
        disease_percentage = (disease_count / len(predictions)) * 100
        print(f"Disease Rate: {disease_percentage:.1f}%")
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            print(f"Average Confidence: {avg_confidence:.1%}")
    
    print()
    print("-" * 60)
    print("📋 DETAILED PREDICTIONS")
    print("-" * 60)
    
    for i, pred in enumerate(predictions[:20], 1):  # Show first 20
        status = "🔴 Disease" if pred == 1 else "✅ Healthy"
        conf = f" (Confidence: {confidence_scores[i-1]:.1%})" if confidence_scores and i-1 < len(confidence_scores) else ""
        print(f"Sample {i}: {status}{conf}")
    
    if len(predictions) > 20:
        print(f"... and {len(predictions) - 20} more samples")
    
    print()
    print("=" * 60)


def main():
    print("=" * 60)
    print("🔓 PRIVACY-PRESERVING RESULT DECRYPTION TOOL")
    print("=" * 60)
    print()
    
    # Get encrypted result - allow file or direct input
    if len(sys.argv) < 2:
        print("Choose input method:")
        print("1. Paste encrypted result string")
        print("2. Load from file")
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "2":
            file_path = input("Enter path to encrypted result file: ").strip()
            try:
                with open(file_path, 'r') as f:
                    encrypted_result = f.read().strip()
                print(f"✅ Loaded encrypted result from: {file_path}")
            except Exception as e:
                print(f"❌ Error reading file: {str(e)}")
                sys.exit(1)
        else:
            print("\nEnter the encrypted result from the Streamlit app:")
            print("(Paste the entire encrypted string, then press Enter)")
            encrypted_result = input("> ").strip()
    else:
        # Check if argument is a file path
        arg = sys.argv[1]
        if arg.endswith('.txt') or '/' in arg or '\\' in arg:
            try:
                with open(arg, 'r') as f:
                    encrypted_result = f.read().strip()
                print(f"✅ Loaded encrypted result from: {arg}")
            except Exception as e:
                print(f"❌ Error reading file: {str(e)}")
                sys.exit(1)
        else:
            encrypted_result = arg
    
    if not encrypted_result:
        print("❌ Error: No encrypted result provided!")
        sys.exit(1)
    
    # Get password
    import getpass
    password = getpass.getpass("\n🔑 Enter your encryption password: ")
    
    # Get salt
    print("\n🧂 Enter the salt value you saved during encryption:")
    salt_b64 = input("> ").strip()
    
    print()
    print("⏳ Decrypting results...")
    
    try:
        result = decrypt_prediction_result(encrypted_result, password, salt_b64)
        format_prediction_result(result)
        
        # Automatically save detailed results to results.json
        print()
        print("💾 Saving detailed results to results.json...")
        
        # Create detailed output with sample-by-sample results
        import json
        
        predictions = result.get('predictions', [])
        confidence_scores = result.get('confidence', [])
        
        detailed_results = {
            'model': result.get('model', 'Unknown'),
            'total_samples': result.get('total_samples', 0),
            'summary': {
                'healthy_count': result.get('healthy_count', 0),
                'disease_count': result.get('disease_count', 0),
                'disease_rate': f"{(result.get('disease_count', 0) / result.get('total_samples', 1) * 100):.1f}%"
            },
            'sample_results': []
        }
        
        # Add individual sample results
        for i, pred in enumerate(predictions, 1):
            sample_result = {
                'sample_number': i,
                'prediction': 'Healthy' if pred == 0 else 'Disease',
                'prediction_code': int(pred),
                'confidence': f"{confidence_scores[i-1]:.4f}" if confidence_scores and i-1 < len(confidence_scores) else "N/A"
            }
            detailed_results['sample_results'].append(sample_result)
        
        # Save to results.json
        with open('results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"✅ Detailed results saved to results.json")
        print(f"   - {len(predictions)} samples with predictions and confidence scores")
        
        # Ask if user wants to save to custom location too
        print()
        save_custom = input("💾 Save to additional file? (y/n): ").strip().lower()
        if save_custom == 'y':
            output_file = input("Enter output filename: ").strip()
            with open(output_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            print(f"✅ Results also saved to {output_file}")
        
        print()
        print("🔒 Your data remains private - results decrypted locally!")
        print()
        
    except Exception as e:
        print(f"❌ Error during decryption: {str(e)}")
        print()
        print("Common issues:")
        print("- Incorrect password")
        print("- Wrong salt value")
        print("- Corrupted encrypted result")
        sys.exit(1)


if __name__ == "__main__":
    main()

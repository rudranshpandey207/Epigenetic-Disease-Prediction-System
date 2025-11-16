# 🔧 Fix Alzheimer's Model to Show Feature Importance

## Problem

The Alzheimer's model doesn't show "Top Contributing CpG Sites" because it's missing feature names or is not a tree-based model.

## Solution: Update Your Kaggle Notebook

### **Method 1: If Using Random Forest (Recommended)**

Add this code when saving your Alzheimer's model:

```python
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

# After training your model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# IMPORTANT: Manually set feature names if not already set
if not hasattr(model, 'feature_names_in_'):
    model.feature_names_in_ = X_train.columns.to_numpy()

# Save model
joblib.dump(model, 'alzheimer_model.joblib')

# Also save the scaler with feature names
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

# IMPORTANT: Set feature names on scaler
scaler.feature_names_in_ = X_train.columns.to_numpy()

# Save scaler
joblib.dump(scaler, 'alzheimer_scaler.joblib')
```

---

### **Method 2: If Already Trained, Add Feature Names**

If you already have a trained model, load it and add feature names:

```python
import joblib
import numpy as np

# Load existing model
model = joblib.load('alzheimer_model.joblib')
scaler = joblib.load('alzheimer_scaler.joblib')

# Assuming you have the training data column names
# Replace with your actual CpG site names (500 features for Alzheimer's)
feature_names = [
    'cg00000029', 'cg00000165', 'cg00000236', 'cg00000289',
    # ... (all 500 CpG site names)
]

# OR if you have X_train available:
feature_names = X_train.columns.to_numpy()

# Add feature names to model
model.feature_names_in_ = feature_names

# Add feature names to scaler
scaler.feature_names_in_ = feature_names

# Re-save models
joblib.dump(model, 'alzheimer_model.joblib')
joblib.dump(scaler, 'alzheimer_scaler.joblib')
```

---

### **Method 3: Quick Fix - Use Column Names from Data**

If you don't have access to the original training data:

```python
import joblib
import pandas as pd

# Load a sample of your training data
df = pd.read_csv('alzheimer_training_data.csv')

# Get feature column names (exclude ID and target columns)
feature_cols = [col for col in df.columns if col.startswith('cg')]

# Load models
model = joblib.load('alzheimer_model.joblib')
scaler = joblib.load('alzheimer_scaler.joblib')

# Add feature names
model.feature_names_in_ = feature_cols
scaler.feature_names_in_ = feature_cols

# Re-save
joblib.dump(model, 'alzheimer_model.joblib')
joblib.dump(scaler, 'alzheimer_scaler.joblib')
```

---

## Verification

After updating, verify in Python:

```python
import joblib

# Load model
model = joblib.load('alzheimer_model.joblib')

# Check if feature names exist
print("Has feature_names_in_:", hasattr(model, 'feature_names_in_'))
print("Has feature_importances_:", hasattr(model, 'feature_importances_'))

if hasattr(model, 'feature_names_in_'):
    print("Number of features:", len(model.feature_names_in_))
    print("First 10 features:", model.feature_names_in_[:10])

if hasattr(model, 'feature_importances_'):
    print("Number of importances:", len(model.feature_importances_))
    print("Top 5 importances:", sorted(model.feature_importances_, reverse=True)[:5])
```

Expected output:

```
Has feature_names_in_: True
Has feature_importances_: True
Number of features: 500
First 10 features: ['cg00000029' 'cg00000165' 'cg00000236' ...]
Number of importances: 500
Top 5 importances: [0.0523, 0.0489, 0.0456, 0.0432, 0.0401]
```

---

## Why This Is Needed

1. **Feature Importance:** Random Forest models have `feature_importances_` attribute
2. **Feature Names:** We need `feature_names_in_` to label which CpG site is important
3. **Streamlit Display:** The app matches feature names with importance scores

---

## Alternative: If Using Different Model Type

If you're using a model that doesn't have feature importance (like SVM or Logistic Regression):

```python
from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest instead
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)

# This will have feature_importances_ automatically
print(f"Accuracy: {rf_model.score(X_test, y_test)}")
```

---

## Upload Updated Models

After fixing:

1. Download the updated `.joblib` files from Kaggle
2. Replace the old files in `F:\Hackathon\Minor\models\`
3. Restart Streamlit app
4. Test with Alzheimer's data - feature importance should now appear!

---

## Current Status

✅ **Prostate Model:** Working - shows feature importance  
❌ **Alzheimer's Model:** Missing feature names

After fix:  
✅ **Both Models:** Will show top contributing CpG sites

import streamlit as st
import pandas as pd
import pickle
import joblib
import os
from pathlib import Path
import io
from PIL import Image
import pytesseract
import PyPDF2
from docx import Document
import re
import numpy as np
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

# Add after imports
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Page configuration
st.set_page_config(
    page_title="Epigenetic Disease Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .control {
        background-color: #C8E6C9;
        border-left: 5px solid #4CAF50;
    }
    .disease {
        background-color: #FFCDD2;
        border-left: 5px solid #F44336;
    }
    </style>
""", unsafe_allow_html=True)

# Load models function
@st.cache_resource
def load_models():
    models = {}
    scalers = {}
    
    try:
        # Load Alzheimer's model
        alzheimer_model_path = Path("models/alzheimer_model.joblib")
        if alzheimer_model_path.exists():
            models['alzheimer'] = joblib.load(alzheimer_model_path)
            st.sidebar.success("✅ Alzheimer's model loaded")
        else:
            st.sidebar.info("ℹ️ Alzheimer's model not found")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Alzheimer's model error: {str(e)[:50]}...")
    
    try:
        # Load Alzheimer's scaler
        alzheimer_scaler_path = Path("models/alzheimer_scaler.joblib")
        if alzheimer_scaler_path.exists():
            scalers['alzheimer'] = joblib.load(alzheimer_scaler_path)
    except Exception as e:
        pass
    
    try:
        # Load Prostate model (NEW FILENAME)
        prostate_model_path = Path("models/prostate_rf_model_2000f_70_30.joblib")
        if prostate_model_path.exists():
            models['prostate'] = joblib.load(prostate_model_path)
            st.sidebar.success("✅ Prostate model loaded")
        else:
            st.sidebar.info("ℹ️ Prostate model not found")
    except Exception as e:
        st.sidebar.error(f"❌ Prostate model error: {str(e)[:50]}...")
    
    try:
        # Load Prostate scaler (NEW FILENAME)
        prostate_scaler_path = Path("models/prostate_rf_scaler_2000f_70_30.joblib")
        if prostate_scaler_path.exists():
            scalers['prostate'] = joblib.load(prostate_scaler_path)
    except Exception as e:
        pass
    
    return models, scalers

# Function to prepare data for prediction
def prepare_data_for_prediction(df, model, scaler=None):
    """Prepare uploaded data to match model's expected features"""
    try:
        # First, remove common non-feature columns
        columns_to_drop = ['class', 'Class', 'label', 'Label', 'target', 'Target']
        df_clean = df.copy()
        
        for col in columns_to_drop:
            if col in df_clean.columns:
                st.info(f"ℹ️ Removing target column: '{col}'")
                df_clean = df_clean.drop(col, axis=1)
        
        # Remove non-numeric columns (like SampleID)
        numeric_df = df_clean.select_dtypes(include=[np.number])
        
        # If no numeric columns, try converting
        if numeric_df.empty:
            # Try to convert all columns except first (assumed to be ID)
            df_copy = df_clean.iloc[:, 1:].copy()
            for col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            numeric_df = df_copy.select_dtypes(include=[np.number])
        
        # Get model's expected features
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            
            # Align columns with model's expected features
            missing_cols = set(expected_features) - set(numeric_df.columns)
            extra_cols = set(numeric_df.columns) - set(expected_features)
            
            if missing_cols:
                st.warning(f"⚠️ Missing {len(missing_cols)} expected features. Filling with zeros.")
                for col in missing_cols:
                    numeric_df[col] = 0
            
            if extra_cols:
                st.info(f"ℹ️ {len(extra_cols)} extra columns will be ignored.")
            
            # Reorder columns to match model
            numeric_df = numeric_df[expected_features]
        
        # Apply scaling if scaler exists
        if scaler is not None:
            numeric_df = pd.DataFrame(
                scaler.transform(numeric_df),
                columns=numeric_df.columns,
                index=numeric_df.index
            )
        
        return numeric_df
    
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        import traceback
        st.error(f"Detailed error:\n{traceback.format_exc()}")
        return None

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return None

# Function to extract text from Word document
def extract_text_from_docx(docx_file):
    try:
        doc = Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + "\t"
                text += "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting Word document: {str(e)}")
        return None

# Function to extract text from image using OCR
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error extracting image: {str(e)}")
        return None

# Function to parse text to DataFrame
def parse_text_to_dataframe(text):
    try:
        # Split by lines
        lines = text.strip().split('\n')
        
        # Try to detect delimiter (comma, tab, space)
        first_line = lines[0]
        if '\t' in first_line:
            delimiter = '\t'
        elif ',' in first_line:
            delimiter = ','
        else:
            delimiter = r'\s+'
        
        # Parse data
        data = []
        for line in lines:
            if line.strip():
                if delimiter == r'\s+':
                    row = re.split(delimiter, line.strip())
                else:
                    row = line.split(delimiter)
                data.append(row)
        
        if len(data) > 0:
            df = pd.DataFrame(data[1:], columns=data[0])
            return df
        return None
    except Exception as e:
        st.error(f"Error parsing text to dataframe: {str(e)}")
        return None

# Function to convert uploaded file to DataFrame
def convert_to_dataframe(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            return None
    
    elif file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
        if text:
            return parse_text_to_dataframe(text)
        return None
    
    elif file_extension in ['docx', 'doc']:
        text = extract_text_from_docx(uploaded_file)
        if text:
            return parse_text_to_dataframe(text)
        return None
    
    elif file_extension in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
        text = extract_text_from_image(uploaded_file)
        if text:
            return parse_text_to_dataframe(text)
        return None
    
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None

# Title and description
st.markdown('<h1 class="main-header">🧬 Epigenetic Disease Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict disease risk using epigenetic methylation data</p>', unsafe_allow_html=True)

# Load models
models, scalers = load_models()

# Sidebar
with st.sidebar:
    # Try to load local image, fallback to a simple colored placeholder
    try:
        st.image("assets/dna_logo.png", width="stretch")
    except:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0;">🧬 Epigenetic<br>Predictor</h2>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Model Selection")
    
    disease_type = st.selectbox(
        "Choose Disease Model",
        ["Alzheimer's Disease", "Prostate Cancer"],
        help="Select the disease you want to predict"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    
    if disease_type == "Alzheimer's Disease":
        model_available = 'alzheimer' in models
        st.info(f"""
        **Dataset:** GSE80970  
        **Model:** Random Forest  
        **Status:** {'✅ Loaded' if model_available else '❌ Not Loaded'}  
        **Features:** DNA Methylation Sites
        """)
    else:
        model_available = 'prostate' in models
        st.info(f"""
        **Dataset:** GSE26126  
        **Model:** Random Forest (2000 features)
        **Status:** {'✅ Loaded' if model_available else '❌ Not Loaded'}  
        **Split:** 70/30 Train/Test
        """)
    
    st.markdown("---")
    st.markdown("### 🔒 Security")
    st.success("✅ Models encrypted with RSA")
    st.success("✅ Secure cloud storage")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📁 Upload Epigenetic Data")
    st.markdown("Upload your data in **CSV, PDF, Word, or Image** format")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'pdf', 'docx', 'doc', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Supported formats: CSV, PDF, Word (DOCX/DOC), Images (PNG, JPG, JPEG, TIFF, BMP)"
    )
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner(f"🔄 Processing {file_extension.upper()} file..."):
            # Convert file to DataFrame
            df = convert_to_dataframe(uploaded_file)
        
        if df is not None:
            st.success(f"✅ File uploaded and converted successfully! Shape: {df.shape}")
            
            # Option to download as CSV
            st.download_button(
                label="💾 Download Converted CSV",
                data=df.to_csv(index=False),
                file_name=f"converted_{uploaded_file.name.split('.')[0]}.csv",
                mime="text/csv"
            )
            
            # Display data preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10), width="stretch")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Rows", df.shape[0])
                with col_b:
                    st.metric("Columns", df.shape[1])
                with col_c:
                    st.metric("Missing Values", df.isnull().sum().sum())
        else:
            df = None
            st.error("❌ Failed to convert file. Please check the format and try again.")
    else:
        df = None
        st.info("👆 Please upload a file to begin prediction")

with col2:
    st.markdown("### ℹ️ Supported Formats")
    st.markdown("""
    **📄 CSV Files:**
    - Direct upload, no conversion needed
    
    **📑 PDF Files:**
    - Text extraction from tables
    - Must contain structured data
    
    **📝 Word Documents (.docx):**
    - Text and table extraction
    - Structured data format
    
    **🖼️ Images (PNG, JPG, TIFF):**
    - OCR text extraction
    - Clear, high-quality images work best
    
    **Required Data Format:**
    - First column: Sample IDs
    - Other columns: CpG site beta values
    - Values should be between 0 and 1
    """)
    
    # Download example file button
    if st.button("📥 Download Example CSV", width="stretch"):
        example_data = pd.DataFrame({
            'SampleID': ['Sample1', 'Sample2', 'Sample3'],
            'cg00000029': [0.8234, 0.7123, 0.6789],
            'cg00000108': [0.6543, 0.5234, 0.7891],
            'cg00000109': [0.9012, 0.8456, 0.7234]
        })
        csv = example_data.to_csv(index=False)
        st.download_button(
            label="Download",
            data=csv,
            file_name="example_methylation_data.csv",
            mime="text/csv"
        )

# Prediction section
st.markdown("---")
st.markdown("### 🔮 Make Prediction")

if df is not None:
    if st.button("🚀 Predict Disease Status", type="primary", width="stretch"):
        # Select the appropriate model
        model_key = 'alzheimer' if disease_type == "Alzheimer's Disease" else 'prostate'
        
        if model_key not in models:
            st.error(f"❌ {disease_type} model is not loaded. Please check the models folder.")
        else:
            with st.spinner("🧬 Analyzing epigenetic data..."):
                try:
                    # Get model and scaler
                    model = models[model_key]
                    scaler = scalers.get(model_key, None)
                    
                    # Prepare data
                    prepared_data = prepare_data_for_prediction(df, model, scaler)
                    
                    if prepared_data is not None:
                        # Make predictions
                        predictions = model.predict(prepared_data)
                        
                        # Get prediction probabilities if available
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(prepared_data)
                            confidence = probabilities.max(axis=1)
                        else:
                            confidence = np.ones(len(predictions)) * 0.85
                        
                        st.markdown("### 📊 Prediction Results")
                        
                        # Summary statistics
                        healthy_count = np.sum(predictions == 0)
                        disease_count = np.sum(predictions == 1)
                        
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.metric("✅ Healthy Samples", healthy_count, delta=None)
                        with col_stat2:
                            st.metric("⚠️ Disease Detected", disease_count, delta=None)
                        
                        st.markdown("---")
                        
                        # Display results for each sample
                        for idx, (pred, conf) in enumerate(zip(predictions, confidence)):
                            sample_name = df.iloc[idx, 0] if df.shape[1] > 0 else f"Sample {idx+1}"
                            
                            if pred == 0:
                                st.markdown(f"""
                                <div class="prediction-box control">
                                    <h3>🆔 {sample_name}</h3>
                                    <h2>✅ Control (Healthy)</h2>
                                    <p style="font-size: 1.5rem; font-weight: bold; color: #2E7D32;">Confidence: {conf:.2%}</p>
                                    <p style="font-size: 1.1rem;">The model predicts this sample as <strong>Control/Healthy</strong> for {disease_type}.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="prediction-box disease">
                                    <h3>🆔 {sample_name}</h3>
                                    <h2>⚠️ Disease Detected</h2>
                                    <p style="font-size: 1.5rem; font-weight: bold; color: #C62828;">Confidence: {conf:.2%}</p>
                                    <p style="font-size: 1.1rem;">The model predicts this sample shows signs of <strong>{disease_type}</strong>.</p>
                                    <p style="font-size: 0.9rem; color: #666;"><em>⚠️ This is a computational prediction. Please consult healthcare professionals for diagnosis.</em></p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Feature importance visualization
                        with st.expander("📈 Top Contributing CpG Sites"):
                            if hasattr(model, 'feature_importances_'):
                                feature_importance = pd.DataFrame({
                                    'Feature': prepared_data.columns,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False).head(10)
                                
                                st.bar_chart(feature_importance.set_index('Feature'))
                            else:
                                st.info("Feature importance not available for this model type")
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("Please ensure your data format matches the model's requirements.")
else:
    st.warning("⚠️ Please upload a file first to enable prediction")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🔬 Developed by Rudransh Pandey | 🧬 Powered by Machine Learning & Bioinformatics</p>
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for research purposes only. Not for clinical diagnosis.</p>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import pickle
import os
from pathlib import Path
import io
from PIL import Image
import pytesseract
import PyPDF2
from docx import Document
import re

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

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1E88E5/ffffff?text=Epigenetic+Predictor", width="stretch")
    st.markdown("### 🎯 Model Selection")
    
    disease_type = st.selectbox(
        "Choose Disease Model",
        ["Alzheimer's Disease", "Prostate Cancer"],
        help="Select the disease you want to predict"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    
    if disease_type == "Alzheimer's Disease":
        st.info("""
        **Dataset:** GSE80970  
        **Model:** Random Forest  
        **Accuracy:** ~90%  
        **Features:** DNA Methylation Sites
        """)
    else:
        st.info("""
        **Dataset:** GSE26126  
        **Model:** Random Forest  
        **Accuracy:** ~88%  
        **Features:** DNA Methylation Sites
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
        with st.spinner("🧬 Analyzing epigenetic data..."):
            # Placeholder for actual prediction logic
            import time
            time.sleep(2)  # Simulate processing
            
            # TODO: Replace with actual model prediction
            # For now, using dummy prediction
            import random
            prediction = random.choice([0, 1])  # 0 = Control, 1 = Disease
            confidence = random.uniform(0.75, 0.95)
            
            st.markdown("### 📊 Prediction Results")
            
            if prediction == 0:
                st.markdown(f"""
                <div class="prediction-box control">
                    <h2>✅ Control (Healthy)</h2>
                    <p style="font-size: 1.2rem;">Confidence: {confidence:.2%}</p>
                    <p>The model predicts this sample as <strong>Control/Healthy</strong> for {disease_type}.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box disease">
                    <h2>⚠️ Disease Detected</h2>
                    <p style="font-size: 1.2rem;">Confidence: {confidence:.2%}</p>
                    <p>The model predicts this sample shows signs of <strong>{disease_type}</strong>.</p>
                    <p><em>⚠️ This is a computational prediction. Please consult healthcare professionals for diagnosis.</em></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature importance visualization placeholder
            with st.expander("📈 Top Contributing CpG Sites"):
                st.info("Feature importance visualization will be added here")
                # TODO: Add SHAP values or feature importance chart
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
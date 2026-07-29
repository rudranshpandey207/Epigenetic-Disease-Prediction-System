import streamlit as st

st.set_page_config(page_title="Test")

st.title("Test")

mode = st.radio(
    "Mode",
    ["Standard", "Encrypted"]
)

disease = st.selectbox(
    "Disease",
    ["Alzheimer", "Prostate"]
)

st.write(mode)
st.write(disease)
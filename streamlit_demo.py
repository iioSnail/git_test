import streamlit as st

st.title("中文拼写纠错（Chinese Spell Correction）")

model_name = st.selectbox("请选择您要使用的模型：", ["MacBert4CSC", "SCOPE"])

@st.cache_resource
def load_model():
    if model_name == "MacBert4CSC":
        pass

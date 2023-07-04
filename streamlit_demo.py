import streamlit as st
import torch

from utils.utils import mock_args, render_color_for_text, compare_text

st.title("中文拼写纠错（Chinese Spell Correction）")

model_name = st.selectbox("请选择您要使用的模型：", ["MacBert4CSC", "SCOPE"])

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_model():
    if model_name == "MacBert4CSC":
        from models.MacBert4CSC import MacBert4CSC_Model
        return MacBert4CSC_Model(mock_args(device=get_device()))

    return None


model = load_model()
if model is None:
    st.error("加载模型失败!")
    st.cache_resource.clear()
    exit(0)


with st.form("csc"):
    sentence = st.text_area("请输入你要修改的文本", max_chars=500)

    submitted = st.form_submit_button("提交")

    if submitted:
        pred = model.predict(sentence)
        diff_index = compare_text(sentence, pred)
        st.text("原句子为：")
        st.markdown(render_color_for_text(sentence, diff_index, color='red', format='markdown'))
        st.text("修改后为：")
        st.markdown(render_color_for_text(pred, diff_index, color='blue', format='markdown'))



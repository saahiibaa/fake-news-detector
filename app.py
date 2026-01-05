import streamlit as st
import joblib

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f2f6ff, #fde7f2);
}

.main-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    color: #3d2c54;
}

.sub-text {
    text-align: center;
    color: #6a4c7a;
    margin-bottom: 30px;
    font-size: 17px;
}

.stTextArea textarea {
    border-radius: 16px;
    font-size: 16px;
    padding: 14px;
    background-color: #ffffff;
    border: 1px solid #d9c6ff;
}

.stButton button {
    background: linear-gradient(135deg, #c6b7ff, #f5a6c9);
    color: #3d2c54;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px 22px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.markdown('<div class="main-title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI to detect misinformation instantly</div>', unsafe_allow_html=True)

news_input = st.text_area("News Article", height=220)

if st.button("🔍 Analyze News"):
    if news_input.strip():
        transformed = vectorizer.transform([news_input])
        prediction = model.predict(transformed)

        if prediction[0] == 1:
            st.markdown('<div class="result-real">✅ This news is REAL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-fake">❌ This news is FAKE</div>', unsafe_allow_html=True)
    else:
        st.warning("Please paste a news article first.")

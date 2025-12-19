import streamlit as st
import pandas as pd
from markov_chain import MarkovChain
import os

# Page Config
st.set_page_config(
    page_title="Nutuk Metin Üretici",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_model():
    return MarkovChain()

try:
    chain = load_model()
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

st.title("🇹🇷 Nutuk Metin Üretici (Markov Zinciri)")
st.markdown("""
Bu uygulama, **Mustafa Kemal Atatürk'ün Nutuk** eserinden eğitilmiş bir Markov Zinciri modeli kullanarak metin üretir.
""")


st.sidebar.header("Ayarlar")

start_word = st.sidebar.text_input("Başlangıç Kelimesi", value="Millet")
length = st.sidebar.slider("Kelime Sayısı (Uzunluk)", min_value=1, max_value=100, value=20)
temperature = st.sidebar.slider("Sıcaklık (Yaratıcılık)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

if st.sidebar.button("Metin Oluştur"):
    with st.spinner("Metin üretiliyor..."):
        generated_text = chain.generate_text(
            start_word=start_word,
            length=length,
            temperature=temperature,
            alpha=0.01 
        )
    
    st.subheader("Üretilen Metin")
    st.success(generated_text)

    st.markdown("---")
    st.subheader("Analiz: Kelime Olasılıkları")
    
    words = generated_text.split()
    if words:
        last_word = words[-1]
        
        st.write(f"**'{last_word}'** kelimesinden sonra gelebilecek en olası kelimeler:")
        
        transitions = chain.get_top_transitions(last_word, top_n=10, alpha=0.01)
        
        if transitions:
            df = pd.DataFrame(list(transitions.items()), columns=["Kelime", "Olasılık"])
            df = df.sort_values(by="Olasılık", ascending=False)
            
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x="Olasılık", y="Kelime", data=df, ax=ax, palette="viridis")
            ax.set_title(f"'{last_word}' Kelimesinden Sonra Gelecek Olası Kelimeler")
            st.pyplot(fig)
        else:
            st.info("Bu kelime için geçiş verisi bulunamadı.")
            
else:
    st.info("Ayarları yapın ve 'Metin Oluştur' butonuna tıklayın.")

st.sidebar.markdown("---")
st.sidebar.info("Veri Kaynağı: NUTUK_1.txt")

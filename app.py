import streamlit as st

from main import process_query

st.set_page_config(
    page_title="Melis Lazer - Yapay Zeka Fiyat Asistanı",
    page_icon="✂️",
    layout="centered",
)

with st.sidebar:
    st.header("Melis Lazer Asistanı")
    st.write(
        "Bu asistan, Melis Lazer'in dokümanlarına dayanarak sorularınızı "
        "yanıtlar ve fiyat teklifi taleplerinizi otomatik olarak hesaplar.\n\n"
        "**Fiyat teklifi almak için** malzeme, en, boy ve adet bilgilerini "
        "belirterek yazmanız yeterlidir, tek bir kalem için:\n"
        "> 50 adet 10x10 pleksi ne kadar tutar?\n\n"
        "birden fazla kalem için ise:\n"
        "> 10 adet açacak ve 20 adet organizer istiyorum\n\n"
        "**Genel sorular için** dokümanlarımızla ilgili herhangi bir şey "
        "sorabilirsiniz."
    )
    st.divider()
    if st.button("Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()

st.title("Melis Lazer - Yapay Zeka Fiyat Asistanı")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Sorunuzu yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum..."):
            try:
                response = process_query(prompt)
            except Exception as e:
                response = f"Bir hata oluştu: {e}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

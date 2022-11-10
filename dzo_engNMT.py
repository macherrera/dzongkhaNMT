import streamlit as st
import translate

header = st.container()
with header:
    st.title('Dzongkha-English translation')

dzo = st.text_area(label='Dzongkha input:', height=200)

button = st.button(label='Translate')

'English output:'

if button:
    st.write(translate.main(dzo))




import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Page: ', ('EDA','Model Prediksi'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()
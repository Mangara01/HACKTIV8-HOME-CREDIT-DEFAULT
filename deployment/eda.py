import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def run_eda(data):

    st.header('Tampilan Data')
    st.write(data.head())
    
    st.header('Statistik Deskriptif')
    st.write(data.describe())
    
    st.header('Jumlah Default')
    default_count = data['TARGET'].value_counts()
    st.write(default_count)
    st.bar_chart(default_count)

    st.header('Distribusi Fitur')
    selected_columns = st.multiselect('Pilih fitur:', data.columns)
    if selected_columns:
        selected_data = data[selected_columns]
        for column in selected_columns:
            plt.figure(figsize=(10, 6))
            if selected_data[column].dtype == 'object':
                sns.countplot(x=column, data=selected_data)
            else:
                sns.histplot(selected_data[column].dropna())
            st.write(column)
            st.pyplot(plt)

def run():
    
    st.title('Home Credit Default - Exploratory Data Analysis')
    st.write('Created By: **Mangara Haposan Immanuel Siagian**')
    image = Image.open('deployment/hc.jpg')
    st.image(image, caption=' ')
    st.markdown('---')
   
    st.write('### Definisi')
    st.write('Home Credit Default merujuk pada situasi di mana peminjam gagal membayar kembali pinjaman yang diberikan oleh lembaga pembiayaan rumah atau kredit.')
    st.markdown('---')
    
    st.write('### Letakkan File disini (max. 200 MB)')
    uploaded_file = st.file_uploader('Unggah file CSV', type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        run_eda(data)

if __name__ == "__main__":
    run()
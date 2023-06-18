import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

with open('deployment/model_rf.pkl', 'rb') as file_1:
    model_rf = joblib.load(file_1)

def transform_categorical_data(data):
    le = LabelEncoder()
    transformed_data = data.copy()
    categorical_columns = transformed_data.select_dtypes(include=['object']).columns

    for column in categorical_columns:
        transformed_data[column] = le.fit_transform(transformed_data[column])

    return transformed_data

def run():
   
    st.title('Prediksi Home Credit Default')

    st.subheader('Input Data')
    with st.form(key='input_form'):
        occupation_type = st.selectbox('Pekerjaan', ['Laborers','Managers','Drivers','Core staff','Sales staff',
                                                      'High skill tech staff','Medicine staff','Accountants','Private service staff',
                                                      'Cooking staff','HR staff','Cleaning staff','Security staff','Secretaries',
                                                      'IT staff','Realty agents','Waiters/barmen staff','Low-skill Laborers'])
        organization_type = st.selectbox('Organisasi', ['Business Entity Type 3','School','Business Entity Type 2','Self-employed',
                                                         'Services','Security','Trade: type 2','Kindergarten','Medicine','Transport: type 2',
                                                         'Government','Other','Trade: type 7','Transport: type 4','Construction','Housing',
                                                         'Hotel','Military','Trade: type 3','Industry: type 9','Postal','Bank','Trade: type 6',
                                                         'Industry: type 2','Police','Industry: type 3','University','Industry: type 1',
                                                         'Industry: type 4','Insurance','Industry: type 7','Legal Services','Industry: type 11',
                                                         'Security Ministries','Electricity','Business Entity Type 1','Transport: type 3',
                                                         'Industry: type 12','Agriculture','Emergency','Restaurant','Mobile','Industry: type 5',
                                                         'Telecom','Advertising','Industry: type 10','Trade: type 5','Realtor'])
        goods_category = st.selectbox('Kategori Barang', ['Vehicles','XNA','Consumer Electronics','Furniture','Clothing and Accessories','Computers',
                                                      'Audio/Video','Mobile','Photo / Cinema Equipment','Construction Materials','Other',
                                                      'Sport and Leisure','Gardening','Office Appliances','Jewelry','Tourism','Auto Accessories',
                                                      'Medical Supplies','Medicine','Fitness','Homewares'])
        education_type = st.selectbox('Edukasi', ['Secondary / secondary special','Higher education','Incomplete higher','Lower secondary',
                                                        'Academic degree'])
        code_gender = st.selectbox('Jenis Kelamin', ['M','F'])
        income_type = st.selectbox('Tipe Income', ['Working','State servant','Commercial associate'])
        yield_group = st.selectbox('Interest Rate', ['low_normal','middle','high','low_action','XNA'])
        family_status = st.selectbox('Status Keluarga', ['Married','Separated','Single / not married','Widow','Civil marriage'])
        portfolio = st.selectbox('Portfolio', ['POS','Cash','Cards'])
        product_type = st.selectbox('Tipe Produk', ['XNA','x-sell','walk-in'])
    
        submit_button = st.form_submit_button(label='PREDIKSI')

    if submit_button:
        
        input_data = pd.DataFrame({
            'OCCUPATION_TYPE': [occupation_type],
            'ORGANIZATION_TYPE': [organization_type],
            'NAME_GOODS_CATEGORY': [goods_category],
            'NAME_EDUCATION_TYPE': [education_type],
            'CODE_GENDER': [code_gender],
            'NAME_INCOME_TYPE': [income_type],
            'NAME_YIELD_GROUP': [yield_group],
            'NAME_FAMILY_STATUS': [family_status],
            'NAME_PORTFOLIO': [portfolio],
            'NAME_PRODUCT_TYPE': [product_type],
        })
    
        transformed_data = transform_categorical_data(input_data)
    
        prediction = model_rf.predict(transformed_data)
    
        result = 'tidak akan Home Credit Default' if prediction[0] == 0 else 'akan Home Credit Default'
    
        st.subheader('Hasil Prediksi:')
        st.write(result)
        
if __name__ == '__main__':
    run()
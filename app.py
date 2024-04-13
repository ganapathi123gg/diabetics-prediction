import streamlit as st
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

sc = pickle.load(open('sc.pkl','rb'))
model = pickle.load(open('lr.pkl','rb'))

st.set_page_config(layout='wide')
st.header('Diabetes Prediction')

preg = st.selectbox('No. of Pregnencies',[0,1,2,3,4,5,6,7,8,9,10])
glucose = st.slider('glucose',50,200)
bp = st.slider('Blood Pressure',80,200)
sst = st.slider('Skin Thickness',0,99)
ins = st.slider('Insulin',0,846)
bmi = st.slider('BMI',0,70)
dp = st.slider('Diabetes Pedigree',0,3)
age = st.slider('Age',10,120)



features=np.array([preg,glucose,bp,sst,ins,bmi,dp,age])

features=features.reshape(1,-1)

features=sc.transform(features)


if st.button('Submit'):
    pred = model.predict(features)

    if pred == 0:
        st.write('No')
    else:
        st.write("Yes")

import pandas as pd
import streamlit as st

df1=pd.read_csv('data.csv')
st.write("Fawaz Imtiaz")
st.markdown('Welcome to my app')
st.write("Breast Cancer Wisconsin Dataset")
st.write(df1)
st.markdown("*")

button=st.radio('Do you want to delete any row having NaN in at least one of the fields', ['No', 'Yes'])
if button=='Yes':
    df=df1.dropna();
    st.write("You deleted rows having NaN in at least one of the fields")
elif button=='No':
    df = df1;

button1=st.button("Show Statistics");
if button1:
    st.write(df.describe())
if st.button("Hide Statistics"):
    button1=False

cols=df.columns
red_df=df.iloc[:,0:32]
red_cols=red_df.columns
button2=st.button("Show Columns");
if button2:
    st.write("No. of columns are ",len(cols))
    st.write("The columns are following-")
    st.write(df.columns)
if st.button("Hide Columns"):
    button2=False
st.write("Please select following variables for different plotting")
xv=st.selectbox('Please select x or first variable:',cols)
yv=st.selectbox('Please select y or second variiable:',cols)
zv=st.selectbox('Please select hue or third variiable:',red_cols)

button3=st.button("Bar Chart");
if button3:
    st.bar_chart(data=df, x=xv, y=yv)

if st.button("Hide Bar Chart"):
    button3=False

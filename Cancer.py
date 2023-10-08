import pandas as pd
import streamlit as st

df1=pd.read_csv('data.csv')
st.write("Fawaz Imtiaz")
st.markdown('Welcome to my app')
st.write("Breast Cancer Wisconsin Dataset")
show_table = st.checkbox("Show Dataset Table")
df2 = df1.dropna(axis=1, how='any')


if show_table:
    st.write(df2)
st.markdown("*")

button=st.radio('Do you want to delete any row having NaN in at least one of the fields', ['No', 'Yes'])
if button=='Yes':
    df=df2.dropna();
    st.write("You deleted rows having NaN in at least one of the fields")
elif button=='No':
    df = df2;

button1=st.button("Show Statistics");
if button1:
    st.write(df.describe())
if st.button("Hide Statistics"):
    button1=False


selected_group = st.radio('Choose a feature group to keep:', ['Worst Features', 'Mean Features', 'Standard Error Features', 'Keep all'])

if selected_group == 'Worst Features':
    df = df[['diagnosis'] + ['id'] + list(df.filter(like='worst'))]
elif selected_group == 'Mean Features':
    df = df[['diagnosis'] + ['id'] + list(df.filter(like='mean'))]
elif selected_group == 'Standard Error Features':
    df = df[['diagnosis'] + ['id'] + list(df.filter(like='se'))]
else:
    pass

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

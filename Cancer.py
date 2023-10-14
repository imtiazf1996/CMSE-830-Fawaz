import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import hiplot as hip
import seaborn as sns
import plotly.express as px

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


selected_group = st.radio('Choose a feature group to keep:', ['Worst Features', 'Mean Features', 'Standard Error Features', 'Keep All'])

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

plot_selection = st.selectbox("Select a plot type:", ["Histogram", "Scatter Plot", "Box Plot", "HiPlot", "Pair Plot", "Violin Plot"])

st.write("Please select following variables for different plotting")
xv=st.selectbox('Please select x or first variable:',cols)
yv=st.selectbox('Please select y or second variiable:',cols)


if plot_selection in ["Histogram", "Scatter Plot", "Box Plot", "HiPlot", "Pair Plot", "Violin Plot"]:
    zv=st.selectbox('Please select hue or third variiable:',red_cols)
else:
    zv = None

if st.button("Generate Plot"):
    if plot_selection == "Histogram":
        st.subheader("Histogram")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=xv, hue=zv, kde=True)
        fig.update_layout(plot_bgcolor="white") 
        st.pyplot(fig)
    
    elif plot_selection == "Scatter Plot":
        st.subheader("Scatter Plot")
        fig = px.scatter(df, x=xv, y=yv, color=zv, title="Scatter Plot")
        fig.update_traces(marker=dict(size=6), selector=dict(mode='markers+text'))
        fig.update_layout(hovermode='closest')
        fig.update_traces(text=df[zv], textposition='top center')
        fig.update_layout(plot_bgcolor="white") 
        st.plotly_chart(fig)

    
    elif plot_selection == "Box Plot":
        st.subheader("Box Plot")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=xv, y=yv, hue=zv)
        fig.update_layout(plot_bgcolor="white") 
        st.pyplot(fig)

    elif plot_selection == "HiPlot":
        st.subheader("HiPlot")
        hiplot_exp = hip.Experiment.from_dataframe(df)
        fig.update_layout(plot_bgcolor="white") 
        st.write(hiplot_exp)

    elif plot_selection == "Pair Plot":
        st.subheader("Pair Plot")
        fig = px.scatter_matrix(df, color=zv, title="Pair Plot")
        fig.update_layout(plot_bgcolor="white")  
        st.plotly_chart(fig)
        
    elif plot_selection == "Violin Plot":
        st.subheader("Violin Plot")
        fig = px.violin(df, x=xv, y=yv, color=zv, title="Violin Plot")
        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig)

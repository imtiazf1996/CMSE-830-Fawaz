import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import hiplot as hip
import seaborn as sns
import plotly.express as px



df1=pd.read_csv('data.csv')
st.write("**Fawaz Imtiaz**")
st.markdown('First Project of CMSE-830')
st.markdown('Welcome to my app')
st.write("**_Breast Cancer Wisconsin Dataset_**")

description=st.checkbox("General information about this project")

if description:
    st.write('In Wisconsin, a dataset was created that gives important information about breast cancer. It comes from the University of Wisconsin Hospitals in Madison. This data helps us understand breast cancer better and is useful for research. Here we will see how the Benign and Malignant type cancer cells are different and how  we can identify them from different properties')
    st.write("In this app, you can: \n1. View and Modify Data: Display the dataset, handle missing values, and select specific feature groups. \n2. Interactive Visualizations: From histograms to scatter plots and heatmaps, delve deep into the data's patterns.\n3. Customize Plots: Choose variables and generate insightful visualizations.")
st.markdown("***")
            
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

plot_selection = st.selectbox("Select a plot type:", ["Histogram", "Scatter Plot", "Pair Plot", "Violin Plot", "3D Scatter Plot", "Correlation Heatmap", "HiPlot"])

st.write("Please select following variables for different plotting")
if plot_selection in ["Histogram", "Scatter Plot", "3D Scatter Plot"]:
    xv=st.selectbox('Please select x :',cols)

if plot_selection in ["Pair Plot"]:
    selected_box= st.multiselect('Select variables:', cols)
    selected_data = df[selected_box + ['diagnosis']]

if plot_selection in ["Correlation Heatmap"]:
    selected_box= st.multiselect('Select variables:', cols)
    selected_data = df[selected_box]

if plot_selection in ["Violin Plot"]:
    xv=df["diagnosis"]

if plot_selection in [ "Scatter Plot", "Violin Plot", "3D Scatter Plot"]:
    yv=st.selectbox('Please select y :',cols)

if plot_selection in ["3D Scatter Plot"]:
    z3=st.selectbox('Please select z:',cols)

st.write("The hue in required plots will be based on Malignant (M) or Benign (B)")
zv='diagnosis'

if st.button("Generate Plot"):
    if plot_selection == "Histogram":
        st.subheader("Histogram")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=xv, hue=zv, kde=True)
        st.pyplot(fig)
    
    elif plot_selection == "Scatter Plot":
        st.subheader("Scatter Plot")
        fig = px.scatter(df, x=xv, y=yv, color=zv, title="Scatter Plot")
        fig.update_traces(marker=dict(size=6), selector=dict(mode='markers+text'))
        fig.update_layout(hovermode='closest')
        fig.update_traces(text=df[zv], textposition='top center')
        st.plotly_chart(fig)

    elif plot_selection == "Pair Plot":
        st.subheader("Pair Plot")
        all_columns = selected_data.columns
        exclude_column = 'diagnosis' 
        dims = [col for col in all_columns if col != exclude_column]
        fig = px.scatter_matrix(selected_data, dimensions = dims, title="Pair Plot", color= 'diagnosis')
        fig.update_layout(plot_bgcolor="white")  
        st.plotly_chart(fig)

    elif plot_selection == "3D Scatter Plot":
        st.subheader("3D Scatter Plot")
        fig = px.scatter_3d(df, x=xv, y=yv, z=z3, color='diagnosis')
        st.plotly_chart(fig)

    elif plot_selection == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        corr_matrix = selected_data.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        plt.title('Correlation Heatmap')
        st.pyplot(fig)

       
    elif plot_selection == "Violin Plot":
        st.subheader("Violin Plot")
        fig = px.violin(df, x=xv, y=yv, color=zv, title="Violin Plot")
        st.plotly_chart(fig)

    elif plot_selection == "HiPlot":
        st.subheader("HiPlot")
        experiment = hip.Experiment.from_dataframe(df)
        st.components.v1.hip_experiment(experiment)
        

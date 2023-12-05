import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import hiplot as hip
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



df1=pd.read_csv('data.csv')

##General Info

st.write("# **_Breast Cancer Wisconsin Dataset_**")

description=st.checkbox("General information about this project")

if description:
    st.write('In Wisconsin, a dataset was created that gives important information about breast cancer. It comes from the University of Wisconsin Hospitals in Madison. This data helps us understand breast cancer better and is useful for research. Here we will see how the Benign and Malignant type cancer cells are different and how  we can identify them from different properties')
    st.write("In this app, you can: \n1. View and Modify Data: Display the dataset and select specific feature groups. \n2. Interactive Visualizations: From histograms to scatter plots and heatmaps, delve deep into the data's patterns.\n3. Customize Plots: Choose variables and generate insightful visualizations. \n4. Predictions: Use the simulator to change any parameter and see the chances of it being a malignant type cancer cell.")
st.markdown("***")
            
show_table = st.checkbox("Show Dataset Table")
df2 = df1.dropna(axis=1, how='any')

Column=st.checkbox("Column of Data available in this Dataset")
if Column:
    st.write("1. id \n\n2.Diagnosis (B=Benign, M=Malignant) \n\n3.Radius \n\n4.Perimeter \n\n5.Texture \n\n6.Area \n\n7.Smoothness \n\n8.Compactness \n\n9.Concavity \n\n10.Concave Points \n\n11.Symmetry \n\n12.Fractal Dimension \n\nEach property have the mean, the standard error, and the worst case")
st.markdown("***")

if show_table:
    st.write(df2)
st.markdown("*")

button=st.radio('Do you want to delete any row having NaN in at least one of the fields', ['No', 'Yes'])
if button=='Yes':
    df=df2.dropna();
    st.write("You deleted rows having NaN in at least one of the fields")
elif button=='No':
    df = df2;
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
button1=st.button("Show Statistics");
if button1:
    st.write(df.describe())
    st.write("Total number of Malignant cases: ", df[df['diagnosis'] == 'M'].shape[0])
    st.write("Total number of Benign cases: ", df[df['diagnosis'] == 'B'].shape[0])

if st.button("Hide Statistics"):
    button1=False

#MEAN COMPARISON
button3=st.button("General Comparison between Malignant and Benign Cells")
if button3:
    
    df3 = df2[['diagnosis'] + ['id'] + list(df2.filter(like='mean'))]
    means = df3.groupby('diagnosis')[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']].mean()
    comparison={}
    for col in means.columns:
        if "mean" in col:
            if means.loc['B', col] > means.loc['M', col]:
                diff = means.loc['B', col] - means.loc['M', col]
                perc = (diff / means.loc['M', col]) * 100
                comparison[col] = f"Benign higher by {perc:.2f}%"
            else:
                diff = means.loc['M', col] - means.loc['B', col]
                perc = (diff / means.loc['B', col]) * 100
                comparison[col] = f"Malignant higher by {perc:.2f}%"
    comp = pd.DataFrame(comparison, index=['Comparison'])
    means = pd.concat([means, comp])
    rename_dict = {col: col.replace("_mean", "") for col in means.columns if "_mean" in col}
    means = means.rename(columns=rename_dict)
    st.table(means)
if st.button("Hide"):
    button3=False
##Plots EDA


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
        fig1, ax = plt.subplots()
        sns.histplot(data=df, x=xv, hue=zv, kde=True)
        st.pyplot(fig1)
    
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
        hp = hip.Experiment.from_dataframe(df)
        hiplot_html = hp.to_html()
        st.components.v1.html(hp.to_html(), height = 800, width = 1600, scrolling=True)

##Classifier

classifier_selection = st.selectbox("Select a classifier type:", ["KNN", "Logistic Regression"])

X = df1.filter(like='mean')
y = df1['diagnosis'].map({'M': 1, 'B': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
input_data = {}
if 'slider_values' not in st.session_state:
    st.session_state['slider_values'] = {feature: float(X[feature].mean()) for feature in X.columns}
for feature in X.columns:
    input_data[feature] = st.slider(f"Adjust {feature.replace('_mean', '')}",float(X[feature].min()),float(X[feature].max()),st.session_state['slider_values'][feature])
    st.session_state['slider_values'][feature] = input_data[feature]
if classifier_selection in ["Logistic Regression"]:
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    st.title("Breast Cancer Diagnosis Simulator")
    input_df = pd.DataFrame([input_data])
    input_df = scaler.transform(input_df)
    prob = clf.predict_proba(input_df)[0][1]
    st.write(f"### **The likelihood of the tumor being malignant is {prob*100:.2f}%.**")

elif classifier_selection in ["KNN"]:
    n_neighbors_val = st.slider("Choose neighbor value", min_value=2, max_value=50, value=25, step=1)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors_val)  
    knn.fit(X_train, y_train)
    input_df = pd.DataFrame([input_data])
    input_df = scaler.transform(input_df)
    knn_prob = knn.predict_proba(input_df)[0][1]
    st.write(f"### **The likelihood of the tumor being malignant with KNN is {knn_prob*100:.2f}%.**")

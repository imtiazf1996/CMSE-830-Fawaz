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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


df=pd.read_csv('data.csv')

##TABS
tabs = ["Home", "Classifier", "PCA", "Linear Regression", "Plots (EDA)", "About Me"]

selected_tab = st.radio("Sections",tabs, index=0)

##General Info

st.write("# **_Breast Cancer Wisconsin Dataset_**")

if selected_tab == "Home":

    st.write("Welcome to the Breast Cancer Diagnosis App!")
    st.write("In collaboration with the University of Wisconsin Hospitals in Madison, we present a comprehensive dataset that offers valuable insights into breast cancer. This dataset is a vital resource for researchers, medical professionals, and anyone interested in understanding breast cancer better.")
    
    # Purpose of the Dataset
    st.write("### Why This Dataset?")
    st.write("Breast cancer is a significant health concern worldwide, and early diagnosis is critical for effective treatment. This dataset contains a wealth of information collected from breast cancer patients, providing a powerful tool for analysis and research. Here's why this dataset is invaluable:")
    
    st.write("1. **Research:** Researchers can explore this dataset to gain a deeper understanding of breast cancer's characteristics, contributing to advancements in diagnosis and treatment.")
    st.write("2. **Early Detection:** Medical professionals can use this dataset to identify patterns and factors associated with malignancy, aiding in early cancer detection.")
    st.write("3. **Educational:** Students and educators can utilize this dataset for educational purposes, learning about data analysis and cancer research.")
    
    # Using the App
    st.write("## How to Use This App?")
    st.write("Our app simplifies the process of working with this dataset. Here's what you can do with it:")
    st.write("### Visualization and Plots")
    st.write("1. **View and Modify Data:** Explore the dataset, select specific feature groups, and make modifications as needed.")
    st.write("2. **Interactive Visualizations:** Dive deep into data patterns using interactive visualizations like histograms, scatter plots, hiplots, and many more.")
    st.write("3. **Customize Plots:** Tailor your visualizations by choosing variables to generate insightful charts. Choose specific variables from the dataset to create customized visualizations. Select independent and dependent variables to analyze relationships effectively.")
    st.write("### Predictions")
    st.write("Empower your decision-making with predictive insights. This app offers powerful prediction capabilities, including:")
    st.write("1. **Classifier Selection:** Choose from a range of classifiers, including Decision Trees, Random Forest, Support Vector Machine (SVM), and more. Select the one that best suits your prediction task.")
    st.write("2. **Principal Component Analysis (PCA):** Apply PCA to reduce dimensionality and visualize data in a transformed space. Understand feature importance and its impact on predictions.")
    st.write("3. **Test and Train Split:** Evaluate the performance of your chosen classifier with test and train data splits. Assess accuracy, precision, recall, and other metrics to make informed decisions.")
    st.write("4. **Parameter Tuning:** Fine-tune classifier parameters to optimize predictive accuracy. The simulator instantly shows how parameter changes affect the likelihood of a cell being malignant.")
    
    # Definitions
    st.write("### What is Benign and Malignant?")
    st.write("In the context of breast cancer diagnosis, it's important to understand the terms:")
    
    st.write("1. **Benign:** A benign tumor is non-cancerous. Cells in a benign tumor do not spread to other parts of the body. They are usually not life-threatening.")
    st.write("2. **Malignant:** A malignant tumor is cancerous. Malignant cells have the potential to invade nearby tissues and spread to other parts of the body, which can be life-threatening if not treated.")
    
    # Aim of the App
    st.write("### Aim of the App")
    st.write("This app has two primary aims:")
    
    st.write("1. **Facilitate Diagnosis:** We aim to assist medical professionals and researchers in diagnosing breast cancer accurately and early by providing easy access to this valuable dataset and predictive tools.")
    st.write("2. **Educational:** We want to promote education and awareness about breast cancer by making this dataset and its analysis accessible to students, educators, and anyone interested in learning about data analysis and cancer research.")
    
    # Get Started
    st.write("### Get Started")
    st.write("Ready to get started? Click on the options below and also at the top of this page to explore the dataset, visualize the data, customize plots, and make predictions. If you have any questions or need assistance, feel free to reach out.")
    
    st.markdown("***")
                
    show_table = st.checkbox("Show Dataset Table")
    df2 = df.dropna(axis=1, how='any')
    
    Column=st.checkbox("Column of Data available in this Dataset")
    if Column:
        st.write("1. id \n\n2.Diagnosis (B=Benign, M=Malignant) \n\n3.Radius \n\n4.Perimeter \n\n5.Texture \n\n6.Area \n\n7.Smoothness \n\n8.Compactness \n\n9.Concavity \n\n10.Concave Points \n\n11.Symmetry \n\n12.Fractal Dimension \n\nEach property have the mean, the standard error, and the worst case")
    st.markdown("***")
    
    if show_table:
        st.write(df2)
    st.markdown("*")
    
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

##Plots
if selected_tab == "Plots (EDA)":
    st.write ("## Visualization and Plots")
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

#Classifier
if selected_tab == "Classifier":
    st.write("## Predictor")
    classifier_selection = st.selectbox("Select a classifier type:", ["KNN", "Logistic Regression", "SVM", "Random Tree", "Gradient Boosting", "Naive Bayes"])
    
    #Confusion Matrix
    def plot_confusion_matrix(cm, classifier_name):
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.title(f'Confusion Matrix for {classifier_name}', size=15)
        st.pyplot(plt)
    if classifier_selection in ["Logistic Regression"]:
        X = df.filter(like='mean')
        y = df['diagnosis'].map({'M': 1, 'B': 0})
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
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        st.title("Breast Cancer Diagnosis Simulator")
        input_df = pd.DataFrame([input_data])
        input_df = scaler.transform(input_df)
        prob = clf.predict_proba(input_df)[0][1]
        st.write(f"### **The likelihood of the tumor being malignant is {prob*100:.2f}%.**")
        y_pred_lr = clf.predict(X_test)
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        st.write(f"Accuracy (Logistic Regression): {accuracy_lr:.2f}")
        st.write(f"F1 Score (Logistic Regression): {f1_lr:.2f}")
        plot_confusion_matrix(cm_lr, "Logistic Regression")
    elif classifier_selection in ["KNN"]:
        X = df.filter(like='mean')
        y = df['diagnosis'].map({'M': 1, 'B': 0})
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
        n_neighbors_val = st.slider("Choose neighbor value", min_value=2, max_value=50, value=25, step=1)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors_val)  
        knn.fit(X_train, y_train)
        input_df = pd.DataFrame([input_data])
        input_df = scaler.transform(input_df)
        knn_prob = knn.predict_proba(input_df)[0][1]
        st.write(f"### **The likelihood of the tumor being malignant with KNN is {knn_prob*100:.2f}%.**")
        y_pred_lr = knn.predict(X_test)
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        st.write(f"Accuracy (KNN): {accuracy_lr:.2f}")
        st.write(f"F1 Score (KNN): {f1_lr:.2f}")
        plot_confusion_matrix(cm_lr, "KNN")
    elif classifier_selection in ["SVM"]:
        X = df.filter(like='mean')
        y = df['diagnosis'].map({'M': 1, 'B': 0})
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
        svm = SVC(probability=True)  
        svm.fit(X_train, y_train)
        input_df = pd.DataFrame([input_data])
        input_df = scaler.transform(input_df)
        svm_prob = svm.predict_proba(input_df)[0][1]
        st.write(f"### **The likelihood of the tumor being malignant with SVM is {svm_prob*100:.2f}%.**")
        y_pred_lr = svm.predict(X_test)
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        st.write(f"Accuracy (SVM): {accuracy_lr:.2f}")
        st.write(f"F1 Score (SVM): {f1_lr:.2f}")
        plot_confusion_matrix(cm_lr, "SVM")
    elif classifier_selection in ["Random Tree"]:
        X = df.filter(like='mean')
        y = df['diagnosis'].map({'M': 1, 'B': 0})
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
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)  # Number of trees can be adjusted
        rf_clf.fit(X_train, y_train)
        input_df = pd.DataFrame([input_data])
        input_df = scaler.transform(input_df)
        rf_prob = rf_clf.predict_proba(input_df)[0][1]
        st.write(f"### **The likelihood of the tumor being malignant with Random Forest is {rf_prob*100:.2f}%.**")
        y_pred_lr = rf_clf.predict(X_test)
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        st.write(f"Accuracy (Random Tree): {accuracy_lr:.2f}")
        st.write(f"F1 Score (Random Tree): {f1_lr:.2f}")
        plot_confusion_matrix(cm_lr, "Random Tree")
    elif classifier_selection in ["Gradient Boosting"]:
        X = df.filter(like='mean')
        y = df['diagnosis'].map({'M': 1, 'B': 0})
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
        gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gb_clf.fit(X_train, y_train)
        input_df = pd.DataFrame([input_data])
        input_df = scaler.transform(input_df)
        gb_prob = gb_clf.predict_proba(input_df)[0][1]
        st.write(f"### **The likelihood of the tumor being malignant with Gradient Boosting is {gb_prob*100:.2f}%.**")
        y_pred_lr = gb_clf.predict(X_test)
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        st.write(f"Accuracy (Gradient Boosting): {accuracy_lr:.2f}")
        st.write(f"F1 Score (Gradient Boosting): {f1_lr:.2f}")
        plot_confusion_matrix(cm_lr, "Gradient Boosting")
    elif classifier_selection in ["Naive Bayes"]:
        X = df.filter(like='mean')
        y = df['diagnosis'].map({'M': 1, 'B': 0})
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
        nb_clf = GaussianNB()
        nb_clf.fit(X_train, y_train)
        input_df = pd.DataFrame([input_data])
        input_df = scaler.transform(input_df)
        nb_prob = nb_clf.predict_proba(input_df)[0][1]
        st.write(f"### **The likelihood of the tumor being malignant with Naive Bayes is {nb_prob*100:.2f}%.**")
        y_pred_lr = nb_clf.predict(X_test)
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        st.write(f"Accuracy (Naive Bayes): {accuracy_lr:.2f}")
        st.write(f"F1 Score (Naive Bayes): {f1_lr:.2f}")
        plot_confusion_matrix(cm_lr, "Naive Bayes")

#PCA
if selected_tab == "PCA":
    st.markdown("## :blue[Principal Component Analysis (PCA)]")
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
    
    selected_features = st.multiselect('Select variables:', cols)
    
    if selected_features:
        # Scaling the selected features
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[selected_features])
    
        # Applying PCA
        pca = PCA(n_components=3)  # Adjust n_components as needed
        principal_components = pca.fit_transform(df_scaled)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    
        # Visualizing the PCA Results
        fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3')
        fig.update_layout(scene=dict(
           xaxis_title='Principal Component 1',
           yaxis_title='Principal Component 2',
           zaxis_title='Principal Component 3'),
           title="PCA of Selected Features")
        st.plotly_chart(fig)
    
        explained_variance = pca.explained_variance_ratio_
        explained_variance_labels = ['PC1', 'PC2', 'PC3']  
        explained_variance_df = pd.DataFrame({'Principal Component': explained_variance_labels, 'Explained Variance': explained_variance})
        st.bar_chart(explained_variance_df, x='Principal Component', y='Explained Variance')

#Linear Regression without train-test split
if selected_tab == "Linear Regression":
    st.write("## Regression Analysis")
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
    # User selects independent and dependent variables
    independent_var = st.selectbox('Select the independent variable', cols)
    dependent_var = st.selectbox('Select the dependent variable', cols)
    
    if st.button('Run Regression without Train-Test Split'):
        plt.figure()
        X = df[[independent_var]]
        y = df[dependent_var]
        model = LinearRegression().fit(X, y)
        
        st.write(f'Coefficient: {model.coef_[0]}')
        st.write(f'Intercept: {model.intercept_}')
        st.write(f'R-squared: {model.score(X, y)}')
    
        # Plot
        plt.scatter(X, y, color='blue')
        plt.plot(X, model.predict(X), color='red')
        plt.xlabel(independent_var)
        plt.ylabel(dependent_var)
        plt.title('Linear Regression Analysis')
        st.pyplot(plt)
    
    # Perform regression with train-test split
    if st.button('Run Regression with Train-Test Split'):
        plt.figure()
        X = df[[independent_var]]
        y = df[dependent_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        st.write(f'Test R-squared: {model.score(X_test, y_test)}')
    
        # Plot for test set
        plt.scatter(X_test, y_test, color='blue')
        plt.plot(X_test, model.predict(X_test), color='red')
        plt.xlabel(independent_var)
        plt.ylabel(dependent_var)
        plt.title('Regression Analysis with Train-Test Split')
        st.pyplot(plt)

if selected_tab == "About Me":
    st.write("## About Me")
    st.write("Hello! My name is Fawaz Imtiaz. I am a Ph.D. student at Michigan State University, specializing in Mechanical Engineering. My passion for cars drives me to explore innovative solutions to complex problems in the field of Internal Combustion Engine.")
    st.write("When I'm not in the lab or working on research, you can find me pursuing my interests and hobbies:")
    st.write("- **Cars:** Related to my research but I always like to explore different types of cars and their engines in my free time. F1 racing is something that really attracts me.")
    st.write("- **Soccer:** I'm a dedicated soccer enthusiast, both as a player and a fan. The beautiful game never ceases to amaze me with its tactics and excitement.")
    st.write("- **Cycling:** I enjoy the freedom of cycling and often take scenic routes to unwind and stay active.")
    st.write("- **Movies:** I have a deep appreciation for films of all genres, from thought-provoking classics to the latest blockbusters. Movie nights are a must for me!")
    
    st.write("Feel free to reach out if you have any questions or want to discuss anything related to mechanical engineering, sports, or cinema.")


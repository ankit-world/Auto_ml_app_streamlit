import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_diabetes, load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

# Page expands to full width
st.set_page_config(page_title='The Machine Learning Algorithm Comparison App',
                   layout='wide')


# ---------------------------------#
# Model building
def build_model(df):
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y
    print(len(Y.unique()))
    if len(Y.unique())>50:

        st.markdown('**1.2. Dataset dimension**')
        st.write('Shape of input variables(X)')
        st.info(X.shape)
        st.write('Shape of target variable(Y)')
        st.info(Y.shape)

        st.markdown('**1.3. Variable details**')
        st.write('Input Feature variables')
        st.info(list(X.columns))
        st.write('Target variable')
        st.info(Y.name)

        # Build lazy model
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, random_state=seed_number)
        reg1 = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
        models_train, predictions_train = reg1.fit(X_train, X_train, Y_train, Y_train)
        reg2 = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
        models_test, predictions_test = reg2.fit(X_train, X_test, Y_train, Y_test)

        st.subheader('**Table of Model Performance**')

        st.write('Training set')
        st.write(predictions_train)
        st.markdown(filedownload(predictions_train, 'training.csv'), unsafe_allow_html=True)

        st.write('Test set')
        st.write(predictions_test)
        st.markdown(filedownload(predictions_test, 'test.csv'), unsafe_allow_html=True)

        st.subheader('**Plot of Model Performance (Test set)**')

        with st.markdown('R-squared'):
            # Vertical
            predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax1 = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test)
            ax1.set(xlim=(0, 1))
        st.markdown(imagedownload(plt, 'plot-r2-vertical.pdf'), unsafe_allow_html=True)
        # Horizontal
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-r2-horizontal.pdf'), unsafe_allow_html=True)

        with st.markdown('**RMSE**'):
            # Vertical
            predictions_test["RMSE"] = [0 if i < 0 else i for i in predictions_test["RMSE"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
        st.markdown(imagedownload(plt, 'plot-rmse-vertical.pdf'), unsafe_allow_html=True)
        # Horizontal
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-rmse-horizontal.pdf'), unsafe_allow_html=True)

        with st.markdown('**Calculation time**'):
            # Vertical
            predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
        st.markdown(imagedownload(plt, 'plot-calculation-time-vertical.pdf'), unsafe_allow_html=True)
        # Horizontal
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-calculation-time-horizontal.pdf'), unsafe_allow_html=True)

    else:

        st.markdown('**1.2. Dataset dimension**')
        st.write('Shape of input variables(X)')
        st.info(X.shape)
        st.write('Shape of target variable(Y)')
        st.info(Y.shape)

        st.markdown('**1.3. Variable details**')
        st.write('Input Feature variables')
        st.info(list(X.columns))
        st.write('Target variable')
        st.info(Y.name)

        # Build lazy model
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, random_state=seed_number)
        clf1 = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)
        models_train, predictions_train = clf1.fit(X_train, X_train, Y_train, Y_train)
        clf2 = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)
        models_test, predictions_test = clf2.fit(X_train, X_test, Y_train, Y_test)

        st.subheader('**Table of Model Performance**')

        st.write('Training set')
        st.write(predictions_train)
        st.markdown(filedownload(predictions_train, 'training.csv'), unsafe_allow_html=True)

        st.write('Test set')
        st.write(predictions_test)
        st.markdown(filedownload(predictions_test, 'test.csv'), unsafe_allow_html=True)

        st.subheader('**Plot of Model Performance (Test set)**')

        with st.markdown('Accuracy'):
            # Vertical
            predictions_test["Accuracy"] = [0 if i < 0 else i for i in predictions_test["Accuracy"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax1 = sns.barplot(y=predictions_test.index, x="Accuracy", data=predictions_test)
            ax1.set(xlim=(0, 1))
        st.markdown(imagedownload(plt, 'plot-Accuracy-vertical.pdf'), unsafe_allow_html=True)
        # Horizontal
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=predictions_test.index, y="Accuracy", data=predictions_test)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-accuracy-Horizontal.pdf'), unsafe_allow_html=True)

        with st.markdown('**ROC-AUC Score**'):
            # Vertical
            predictions_test["ROC AUC"] = [0 if i < 0 else i for i in predictions_test["ROC AUC"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax2 = sns.barplot(y=predictions_test.index, x="ROC AUC", data=predictions_test)
        st.markdown(imagedownload(plt, 'plot-roc_auc_score-vertical.pdf'), unsafe_allow_html=True)
        # Horizontal
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=predictions_test.index, y="ROC AUC", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-roc_auc_score-horizontal.pdf'), unsafe_allow_html=True)

        with st.markdown('**F1 Score**'):
            # Vertical
            predictions_test["F1 Score"] = [0 if i < 0 else i for i in predictions_test["F1 Score"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax2 = sns.barplot(y=predictions_test.index, x="F1 Score", data=predictions_test)
        st.markdown(imagedownload(plt, 'plot-f1_score-vertical.pdf'), unsafe_allow_html=True)
        # Horizontal
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=predictions_test.index, y="F1 Score", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-f1_score-horizontal.pdf'), unsafe_allow_html=True)

        with st.markdown('**Calculation time**'):
            # Vertical
            predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
        st.markdown(imagedownload(plt, 'plot-calculation-time-vertical.pdf'), unsafe_allow_html=True)
        # Horizontal
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-calculation-time-horizontal.pdf'), unsafe_allow_html=True)



# Download CSV data
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href


def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href



st.write("""
# The Machine Learning Algorithm Comparison App
In this implementation, the **lazypredict** library is used for building several machine learning models at once.

Developed by: Ankit Marwaha
""")

st.write("[source code](https://github.com/ankit-world/Auto_ml_app_streamlit)")


# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file(Regression)](https://raw.githubusercontent.com/ankit-world/Auto_ml_app_streamlit/main/data.csv)
""")

    st.sidebar.markdown("""
[Example CSV input file(Classification)](https://raw.githubusercontent.com/ankit-world/Auto_ml_app_streamlit/main/data.csv)
""")


# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)

# ---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Boston housing dataset
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y = pd.Series(boston.target, name='response')
        df = pd.concat([X, Y], axis=1)

        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)

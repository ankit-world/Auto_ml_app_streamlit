import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.datasets import load_diabetes, load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

# Page extends to full width
st.set_page_config(page_title="The Machine Learning Comparison APP", layout="wide")


# Model Building
def build_model(df):
    df = df.loc[:100]
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    st.markdown('**1.2. Dataset Dimension')
    st.write('x')
    st.info(x.shape)
    st.write('y')
    st.info(y.shape)

    st.markdown('**1.3. Variable Details')
    st.write('x variable (show first 20 rows')
    st.info(list(x.columns))
    st.write('y variable')
    st.info(y.name)

    # Build lazy model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_size, random_state=seed_number)
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models_train, predictions_train = reg.fit(x_train, x_train, y_train, y_train)
    models_test, predictions_test = reg.fit(x_train, x_test, y_train, y_test)

    st.subheader('2. Table of Model Performance')

    st.write('Training set')
    st.write(predictions_train)
    st.markdown(filedownload(predictions_train, 'training.csv'), unsafe_allow_html=True)

    st.write('Test set')
    st.write(predictions_test)
    st.markdown(filedownload(predictions_test, 'test.csv'), unsafe_allow_html=True)

    st.subheader('3. Plot of model Performance (Test set)')

    with st.markdown('**R-squared'):
        # Tall
        predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"]]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test)

    st.markdown(imagedownload(plt, 'plot-r2-tall.pdf'), unsafe_allow_html=True)

    # wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax1 = sns.barplot(x=predictions_train.index, y="R-Squared", data=predictions_train)
    ax1.set(ylim=(0, 1))
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt, 'plot-r2-wide.pdf'), unsafe_allow_html=True)

    with st.markdown('**RMSE (capped at 50)**'):
        # Tall
        predictions_test["RMSE"] = [50 if i > 50 else i for i in predictions_test["RMSE"]]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)

    st.markdown(imagedownload(plt, 'plot-rmse-tall.pdf'), unsafe_allow_html=True)

    # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt, 'plot-rmse-wide.pdf'), unsafe_allow_html=True)

    with st.markdown('**Calculation time**'):
        # Tall
        predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"]]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)

    st.markdown(imagedownload(plt, 'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)

    # wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax3 = sns.barplot(x=predictions_train.index, y="Time Taken", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt, 'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)

    with st.markdown('**RMSE (capped at 50)**'):
        # Tall
        predictions_test["RMSE"] = [50 if i > 50 else i for i in predictions_test["RMSE"]]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)

    st.markdown(imagedownload(plt, 'plot-rmse-tall.pdf'), unsafe_allow_html=True)

    # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt, 'plot-rmse-wide.pdf'), unsafe_allow_html=True)


# Download CSV data

def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href


def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href


st.write("""
# The Machine Learning Algorithm Comparison App

In this implementation, the **lazypredict** library is used for building several machine learning model at once.

Developed by : [Ankit Marwaha]

""")

# Collects user input features into dataframes
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.git
    
    """)

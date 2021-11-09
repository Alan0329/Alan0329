# https://github.com/dataprofessor/model_performance_app/blob/main/app.py
# https://www.youtube.com/watch?v=Ge17mZe54dY&list=PLRwlRpYDDfLCCfZSL4Co9RChjO68a8Vc4&index=13&t=684s

from sklearn import metrics
import streamlit as st
import pandas as pd
import base64
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, cohen_kappa_score


def calc_metrics(input_data):
    y_actual = input_data.iloc[:, 0]
    y_predicted = input_data.iloc[:, 1]
    acc = accuracy_score(y_actual, y_predicted)
    balanced_accuracy = balanced_accuracy_score(y_actual, y_predicted)
    precision = precision_score(y_actual, y_predicted, average='weighted')
    recall = recall_score(y_actual, y_predicted, average='weighted')
    mcc = matthews_corrcoef(y_actual, y_predicted)
    f1 = f1_score(y_actual, y_predicted, average='weighted')

    acc_series = pd.Series(acc, name="Accuracy")
    balanced_accuracy_score_series = pd.Series(
        balanced_accuracy, name="balanced_accuracy_score")
    precision_series = pd.Series(precision, name="precision")
    recall_series = pd.Series(recall, name="recall")
    mcc_series = pd.Series(mcc, name="mcc")
    f1_series = pd.Series(f1, name="f1")

    df = pd.concat(
        [acc_series, balanced_accuracy_score_series, precision_series, recall_series, mcc_series, f1_series], axis=1)

    return df


def calc_cm(input_data):
    y_actual = input_data.iloc[:, 0]
    y_predicted = input_data.iloc[:, 1]
    cm_array = confusion_matrix(y_actual, y_predicted)
    cm_df = pd.DataFrame(cm_array, columns=['Actual', 'Predicted'], index=[
                         'Actual', 'Predicted'])
    return cm_df


def load_example_data():
    
    df = pd.read_csv('https://github.com/Alan0329/Alan0329/raw/main/Other/streamlit_deploy/Y_example.csv')
    return df


def filedownload(df):
    csv = df.to_csv(index=False)
    # strings <-> bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="performance_metrics.csv">Download CSV File</a>'
    return href


st.sidebar.header("Input panel")
st.sidebar.markdown("""
[Example CSV file](https://raw.githubusercontent.com/dataprofessor/model_performance_app/main/Y_example.csv)
""")

upload_file = st.sidebar.file_uploader("上傳檔案", type=['csv'])

select_metrics = ["Accuracy", "balanced_accuracy_score",
                  'precision', 'recall', 'mcc', 'f1']
select_metrics = st.sidebar.multiselect("性能評估選擇", select_metrics)

st.title("Model Performance APP")

if upload_file is not None:
    input_df = pd.read_csv(upload_file)
    cm_df = calc_cm(input_df)
    metrics_df = calc_metrics(input_df)
    selected_metircs_df = metrics_df[select_metrics]
    st.header('Input data')
    st.write(input_df)
    st.header('Confusion matrix')
    st.write(cm_df)
    st.header('Performance metrics')
    st.write(selected_metircs_df)
    st.markdown(filedownload(selected_metircs_df), unsafe_allow_html=True)
else:
    st.info('Awaiting the upload of the input file.')
    if st.button('Use Example Data'):
        input_df = load_example_data()
        confusion_matrix_df = calc_cm(input_df)
        metrics_df = calc_metrics(input_df)
        selected_metrics_df = metrics_df[select_metrics]
        st.header('Input data')
        st.write(input_df)
        st.header('Confusion matrix')
        st.write(confusion_matrix_df)
        st.header('Performance metrics')
        st.write(selected_metrics_df)
        st.markdown(filedownload(selected_metrics_df), unsafe_allow_html=True)

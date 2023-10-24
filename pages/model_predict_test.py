import streamlit as st
import pandas as pd
import numpy as np
import mlflow, json, math, joblib, time
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
from streamlit_session_state import SessionState




st.session_state['run_id']= ''
st.seeeion_state['model_option']= 'Logistic'
def model_on_change():
    st.session_state['run_id']= ''


def app():
    df_file = st.file_uploader("Choose 'gender' file :", key="gender_file_upload")
    df = None
    if df_file is not None:
        df = pd.read_csv(df_file)
        with st.expander(f"check dataset size {df.shape}"):
            col_data, col_pie = st.columns(2)
            with col_data:
                first_name_option = st.selectbox('Chose FirstName Column', df.columns.values, index=df.columns.get_loc('first_name') if 'first_name' in df.columns.values else 0)
                gender_option = st.selectbox('Chose Gender Column', df.columns.values, index=df.columns.get_loc('gender') if 'gender' in df.columns.values else 0)
                df[first_name_option] = df[first_name_option].fillna('')
                df['gender_code'] = df[gender_option].map({'M': 1, 'F': 0})
                df = df[~df['gender_code'].isna()]
                distribution = df[first_name_option].value_counts()
                st.bar_chart(distribution)
            with col_pie:
                fig, ax = plt.subplots()
                num_boys, num_girls = df[df[gender_option] == 'M'].shape[0], df[df[gender_option] == 'F'].shape[0]
                ax.pie([num_girls, num_boys], labels=['Female', 'Male'], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)
                st.write(f'Male: :blue[{num_boys}] and Female: :blue[{num_girls}]')
    else:
        df = None
    model_options = st.selectbox('Choose Model', ('Logistic', 'NN', 'CatBoost'), key='model_options')
    if st.seeeion_state['model_option'] != model_options:
        st.seeeion_state['model_option'] = model_options
        model_on_change()
    run_id = st.text_input('RUN ID', st.seeeion_state['run_id'])
    
    if model_options == 'Logistic':
        if run_id == '' and df is not None:
            st.info(f"Please type in RUN ID and upload the predict data")
        else:
            classifier = mlflow.sklearn.load_model(f"runs:/{run_id}/logistic_gender.pkl")
            vectorizer = mlflow.sklearn.load_model(f"runs:/{run_id}/countvectorizer_gender.pkl")
            X_test, y_test = df[[first_name_option]], df[['gender_code']]
            X_test_vectorized = vectorizer.transform(X_test[first_name_option].values)
            y_pred = classifier.predict(X_test_vectorized)
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
    
    
    if model_options == 'CatBoost':
        if run_id == '' and df is not None:
            st.info(f"Please type in RUN ID and upload the predict data")
        else:
            catB_model = mlflow.catboost.load_model(f"runs:/{run_id}/catboost_model")
            df_ = df.rename(columns={first_name_option : 'first_name'})
            X_test, y_test = df_[['first_name']], df_[['gender_code']]
            cat_features = ['first_name']
            test_data = Pool(data=X_test, cat_features=cat_features)
            y_pred = catB_model.predict(test_data)
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)




if __name__ == '__main__':
    st.markdown('# Gender Classification Model - Predict & Test')
    app()
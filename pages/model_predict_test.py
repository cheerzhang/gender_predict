import streamlit as st
import pandas as pd
import numpy as np
import mlflow, json, math, joblib, time



def app():
    model_options = st.selectbox('Chose Model', ('Logistic', 'NN', 'CatBoost',  'XGB'))
    if model_options == 'Logistic':
        run_id = st.text_input('RUN ID', 'xxx')
        model1 = mlflow.sklearn.load_model(f"runs:/{run_id}/logistic_gender.pkl")
        model2 = mlflow.sklearn.load_model(f"runs:/{run_id}/countvectorizer_gender.pkl")
        st.write(model1)
        st.write(model2)


if __name__ == '__main__':
    app()
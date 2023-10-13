import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import mlflow
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import xgboost as xgb
import catboost
from catboost import CatBoostClassifier, Pool


def log_mdoel(model_name, model, result, experiment_name = 'Gender'):
    with open('config.json') as config_file:
        config = json.load(config_file)
    model_uri = config['tracking_uri']
    mlflow.set_tracking_uri(model_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment = mlflow.create_experiment(name=experiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        # Log parameters
        mlflow.log_params({'name': model_name})
        st.write(result)
        mlflow.log_params(result)
        mlflow.sklearn.log_model(sk_model=model, artifact_path=model_name)
        mlflow.end_run()
        return f"Log model - {model_name} succeed"


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


def app():
    st.markdown('# Gender Classification Model')
    df_file = st.file_uploader("Choose 'gender' file :", key="gender_file_upload")
    if df_file is not None:
        df = pd.read_csv(df_file)
        st.write(df.shape)
        with st.expander(f"check dataset size {df.shape}"):
            col_data, col_pie = st.columns(2)
            with col_data:
                df['gender'] = df['gender_code'].map({'M': 1, 'F': 0, 'U': 2})
                st.dataframe(df)
            with col_pie:
                fig, ax = plt.subplots()
                num_boys, num_girls, num_unknown = df[df['gender_code'] == 'M'].shape[0], df[df['gender_code'] == 'F'].shape[0], df[df['gender_code'] == 'U'].shape[0]
                ax.pie([num_girls, num_boys, num_unknown], labels=['Female', 'Male', 'Unknown'], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)
                st.write(f"female count: :green[{num_girls}]")
                st.write(f"male count: :green[{num_boys}]")
                st.write(f"unknown count: :green[{num_unknown}]")
        # Split the dataset into training and testing sets (80% train, 20% test)
        # some fe
        df['len_first_name'] = df['first_name'].apply(len)
        X = df[['first_name', 'len_first_name']]
        y = df['gender']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Feature extraction using CountVectorizer
        vectorizer = CountVectorizer()
        X_train_vectorized = vectorizer.fit_transform(X_train['first_name'].values)
        X_test_vectorized = vectorizer.transform(X_test['first_name'].values)
        X_train_features = pd.DataFrame(X_train_vectorized.toarray(), columns=vectorizer.get_feature_names_out())
        X_train_features['len_first_name'] = X_train['len_first_name']
        X_test_features = pd.DataFrame(X_test_vectorized.toarray(), columns=vectorizer.get_feature_names_out())
        X_test_features['len_first_name'] = X_test['len_first_name']
        X_train_features = X_train_features.dropna()
        X_test_features = X_test_features.dropna()
        y_train_ = y_train[X_train_features.index]
        y_test_ = y_test[X_test_features.index]
        options = st.selectbox('Chose Model', ('CatBoost', 'Logistic+OVR',  'XGB'))
        if options == 'CatBoost':
            with st.spinner(f"traning CatBoost"):
                classifier = CatBoostClassifier(iterations=500,  # Number of boosting iterations
                                depth=6,         # Tree depth
                                learning_rate=0.1,  # Learning rate
                                loss_function='MultiClass',  # Specify the loss function for multi-class
                                cat_features=[0]  # List of indices of categorical features (if applicable)
                                )
                classifier.fit(X_train[['first_name', 'len_first_name']], y_train)
                y_pred = classifier.predict(X_test[['first_name', 'len_first_name']])
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
        if options == 'Logistic+OVR':
            with st.spinner(f"Logistic+OVR"):
                classifierSearch = LogisticRegression(multi_class='ovr', solver='liblinear', class_weight='balanced')
                param_grid = {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2']
                }
                grid_search = GridSearchCV(classifierSearch, param_grid, cv=10, scoring='f1_weighted')
                grid_search.fit(X_train_features, y_train_)
                st.write(grid_search.best_estimator_)
                classifier = grid_search.best_estimator_
                classifier.fit(X_train_features, y_train_)
                y_pred = classifier.predict(X_test_features)
                # Evaluate the model
                report = classification_report(y_test_, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
        if options == 'XGB':
            model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss")
            model.fit(X_train_features.values, y_train_)
            y_pred = model.predict(X_test_features)
            report = classification_report(y_test_, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        
        # download model
        '''
        with col_option1:
            if st.button('Log logistic_gender Model'):
                msg = log_mdoel('logistic_gender.pkl', obj_model.model,  {'accuracy': accuracy_te, 
                                                                            'recall': recall_te, 
                                                                            'precesion': precision_te,
                                                                            'f1': f1_te})
                joblib.dump(obj_model.model, 'models/logistic_gender.pkl')
                st.success(msg)
        with col_option2:
            if st.button('Log CountVectorizer Model'):
                msg = log_mdoel('countvectorizer_gender.pkl', obj_model.vectorizer, {'accuracy': accuracy_te, 
                                                                            'recall': recall_te, 
                                                                            'precesion': precision_te,
                                                                            'f1': f1_te})
                joblib.dump(obj_model.vectorizer, 'models/countvectorizer_gender.pkl')
                st.success(msg)
        '''




       



if __name__ == '__main__':
    app()
import streamlit as st
import pandas as pd
import numpy as np
import mlflow, json, math, joblib, time
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset



letter_to_number = {'a': 1,  'b': 2,  'c': 3,  'd': 4,  'e': 5,  'f': 6,  'g': 7,  'h': 8,  'i': 9,  'j': 10, 
                    'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 
                    'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, 
                    'A': 27, 'B': 28, 'C': 29, 'D': 30, 'E': 31, 'F': 32, 'G': 33, 'H': 34, 'I': 35, 'J': 36, 
                    'K': 37, 'L': 38, 'M': 39, 'N': 40, 'O': 41, 'P': 42, 'Q': 43, 'R': 44, 'S': 45, 'T': 46, 
                    'U': 47, 'V': 48, 'W': 49, 'X': 50, 'Y': 51, 'Z': 52, 
                    '.': 53, '-': 54, ' ': 55, '@': 56, '?': 57, '/': 58,  "'": 59}

def encode_name(name):
    encoded_name = [letter_to_number[letter] for letter in name if letter in letter_to_number]
    return encoded_name


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def download_csv(df, file_name):
    csv = convert_df(df)
    st.download_button(
        label=f"Download {file_name} CSV",
        data=csv,
        file_name=f'[Gender]{file_name}.csv',
        mime='text/csv',
    )

def app():
    df_file = st.file_uploader("Choose 'gender' file :", key="gender_file_upload")
    df = None
    with open('config.json') as config_file:
        config = json.load(config_file)
    model_uri = config['tracking_uri']
    mlflow.set_tracking_uri(model_uri)
    if df_file is not None:
        try:
            df = pd.read_csv(df_file)
        except (pd.errors.ParserError, UnicodeDecodeError):
            df = pd.read_excel(df_file)
        except Exception as e:
            st.write(e)
        with st.expander(f"check dataset size {df.shape}"):
            col_data, col_pie = st.columns(2)
            with col_data:
                first_name_option = st.selectbox('Chose FirstName Column', df.columns.values, index=df.columns.get_loc('first_name') if 'first_name' in df.columns.values else 0)
                gender_option = st.selectbox('Chose Gender Column', df.columns.values, index=df.columns.get_loc('gender') if 'gender' in df.columns.values else 0)
                df.drop_duplicates(subset=first_name_option, keep='first', inplace=True)
                st.write(df[first_name_option].nunique())
                df[first_name_option] = df[first_name_option].fillna('')
                gender_map = {'M': 1, 'F': 0}
                df['gender_code'] = -1
                df['gender_code'] = df[gender_option].map(gender_map)
                # df = df[~df['gender_code'].isna()]
                st.dataframe(df)
                distribution = df[first_name_option].value_counts()
                st.write(distribution)
                # st.bar_chart(distribution)
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
    run_id = st.text_input('RUN ID', '', key='run_id_text')
    
    if model_options == 'Logistic':
        if run_id == '' or df is None:
            st.info(f"Please type in RUN ID and upload the predict data")
        else:
            with st.spinner(f"Loading {model_options} Model and Predicting..."):
                classifier = mlflow.sklearn.load_model(f"runs:/{run_id}/logistic_gender.pkl")
                vectorizer = mlflow.sklearn.load_model(f"runs:/{run_id}/countvectorizer_gender.pkl")
                X_test, y_test = df[[first_name_option]], df[['gender_code']]
                X_test_vectorized = vectorizer.transform(X_test[first_name_option].values)
                y_pred = classifier.predict(X_test_vectorized)
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
    
    
    if model_options == 'CatBoost':
        if run_id == '' or df is None:
            st.info(f"Please type in RUN ID and upload the predict data")
        else:
            with st.spinner(f"Loading {model_options} Model and Predicting..."):
                catB_model = mlflow.catboost.load_model(f"runs:/{run_id}/catboost_model")
                df_ = df.rename(columns={first_name_option : 'first_name'})
                X_test, y_test = df_[['first_name']], df_[['gender_code']]
                cat_features = ['first_name']
                test_data = Pool(data=X_test, cat_features=cat_features)
                y_pred = catB_model.predict(test_data)
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

    

    if model_options == 'NN':
        if run_id == '' or df is None:
            st.info(f"Please type in RUN ID and upload the predict data")
        else:
            with st.spinner(f"Loading {model_options} Model and Predicting..."):
                NNmodel = mlflow.pytorch.load_model(f"runs:/{run_id}/NN_Transformer")
                client = mlflow.tracking.MlflowClient()
                run = client.get_run(run_id)
                params = run.data.params
                max_name_length = int(params.get('max_name_length'))
                batch_size = int(params.get('batch size'))

                df['encoded_names'] = df[first_name_option].apply(lambda name: encode_name(name))  
                test_sequences = df['encoded_names'].tolist()
                test_labels = df['gender_code'].values
                for i in range(len(test_sequences)):
                    if len(test_sequences[i]) < max_name_length:
                        test_sequences[i] = test_sequences[i] + [0] * (max_name_length - len(test_sequences[i]))
                    elif len(test_sequences[i]) > max_name_length:
                        test_sequences[i] = test_sequences[i][:max_name_length]
                test_sequences = torch.LongTensor(test_sequences)
                test_labels = torch.LongTensor(test_labels)
                test_dataset = TensorDataset(test_sequences, test_labels)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                all_predictions = []
                all_true_labels = []
                NNmodel.eval()
                with torch.no_grad():
                    for val_sequences, val_labels in test_loader:
                        val_output = NNmodel(val_sequences)
                        val_probs = torch.softmax(val_output, dim=1)
                        val_preds = torch.argmax(val_probs, dim=1)
                        item_preds = [item for item in val_preds.tolist()]
                        all_predictions = all_predictions + item_preds
                        all_true_labels = all_true_labels + val_labels.tolist()
                # assign the result back
                st.write(len(all_predictions))
                df['pred_gender'] = all_predictions
                df_remove_duplicate = df[[first_name_option, 'pred_gender']]
                df_remove_duplicate.drop_duplicates(subset=first_name_option, keep='first', inplace=True)
                st.dataframe(df_remove_duplicate)
                download_csv(df_remove_duplicate, 'gender_with_pred')
                # end
                report = classification_report(all_true_labels, all_predictions, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)




if __name__ == '__main__':
    st.markdown('# Gender Classification Model - Predict & Test')
    app()
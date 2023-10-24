import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import mlflow, json, math, joblib, time
import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from catboost import CatBoostClassifier, Pool



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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, sequence_length = 38):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model * sequence_length, 3)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        B, _, _ = output.shape
        output = output.reshape(B, -1)
        output = self.linear(output) # B ,S * D
        return output


def log_mdoel(model_name, model, result, data_size, experiment_name = 'Gender', model_type='NN', model_parameter = None):
    with open('config.json') as config_file:
        config = json.load(config_file)
    model_uri = config['tracking_uri']
    mlflow.set_tracking_uri(model_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment = mlflow.create_experiment(name=experiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        # Log parameters
        st.write(result)
        mlflow.set_tag("run_id", run.info.run_id)
        mlflow.set_tag("data_size", data_size['data size'])
        mlflow.set_tag("data_source", data_size['data source'])
        mlflow.log_metric("f1-score", round(result['f1-score'], 2))
        mlflow.log_metric("precision", round(result['precision'], 3) * 100)
        mlflow.log_metric("recall", round(result['recall'], 3) * 100)
        mlflow.log_metric("support", result['support'])

        if model_type == 'NN':
            mlflow.pytorch.log_model(model, model_name)
            mlflow.log_params(model_parameter)
        if model_type == 'Logistic':
            mlflow.sklearn.log_model(sk_model=model[0], artifact_path=model_name[0])
            mlflow.sklearn.log_model(sk_model=model[1], artifact_path=model_name[1])
        if model_type == 'CatBoost':
            mlflow.catboost.log_model(model, "catboost_model")
        mlflow.end_run()
        return f"Log model - {model_name} succeed"


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


def format_duration(duration):
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        return f"{hours}h {minutes}min {seconds}sec"


def app():
    st.markdown('# Gender Classification Model')
    df_file = st.file_uploader("Choose 'gender' file :", key="gender_file_upload")
    data_source = st.text_input('Data Scorce', 'Public Dataset')
    if df_file is not None:
        df = pd.read_csv(df_file)
        # df = df.iloc[0:30000]
        with st.expander(f"check dataset size {df.shape}"):
            col_data, col_pie = st.columns(2)
            with col_data:
                first_name_option = st.selectbox('Chose FirstName Column', df.columns.values, index=df.columns.get_loc('first_name') if 'first_name' in df.columns.values else 0)
                initials_option = st.selectbox('Chose Initals Column (Chose False to disable use initals for feature)', df.columns.values.tolist()+['False'], index=df.columns.get_loc('initials') if 'initials' in df.columns.values else df.columns.get_loc('False'))
                gender_option = st.selectbox('Chose Gender Column', df.columns.values, index=df.columns.get_loc('gender') if 'gender' in df.columns.values else 0)
                df[first_name_option] = df[first_name_option].fillna('')
                df[gender_option] = df[gender_option].fillna('')
                df['gender_code'] = df[gender_option].map({'M': 1, 'F': 0})
                st.write(df.shape)
                df_name = df[[first_name_option, gender_option, 'gender_code']]
                if initials_option is not 'False':
                    df_initals = df[[initials_option, gender_option, 'gender_code']]
                    df_initals_ = df_initals.rename(columns={initials_option : first_name_option})
                    df_ = pd.concat([df_name, df_initals_])
                    st.write(df_.shape)
                    df = df_
                st.dataframe(df)
                df = df[~df['gender_code'].isna()]
                df = df[df[first_name_option].str.len() >= 3]
            with col_pie:
                fig, ax = plt.subplots()
                num_boys, num_girls = df[df[gender_option] == 'M'].shape[0], df[df[gender_option] == 'F'].shape[0]
                ax.pie([num_girls, num_boys], labels=['Female', 'Male'], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)
                st.write(f'Male: :blue[{num_boys}] and Female: :blue[{num_girls}]')
        
        # chose the model to train
        train_data, val_data = train_test_split(df[[first_name_option, 'gender_code']], test_size=0.2, random_state=42)
        data_size = {'data size': df.shape[0], 
                     'data source': data_source}
        model_options = st.selectbox('Chose Model', ('Logistic', 'NN', 'CatBoost',  'XGB'))
        if model_options == 'NN':
            col_parameter, col_display_parameter = st.columns(2)
            with col_parameter:
                batch_size = st.number_input('Batch Size', min_value = 16, max_value = 128, value = 32)
                embedding_dim = st.number_input('Embedding Size', min_value = 16, max_value = 128, value = 16)
                nhead = st.number_input('N head', min_value = 1, max_value = 128, value = 8)
                d_hid = st.number_input('Hidden Size', min_value = 8, max_value = 128, value = 32)
                nlayers = st.number_input('Layers', min_value = 1, max_value = 10, value = 3)
                dropout = st.number_input('Dropout', min_value = 0.0, max_value = 1.0, value = 0.5)
            with col_display_parameter:
                num_classes = st.number_input('Class Number', value = 2, disabled=True)
                vocab_size =  st.number_input('Vocab Size', value = len(letter_to_number)+1, disabled=True)
            with col_display_parameter:
                model_parameter = {
                    'batch size': batch_size, 
                      'classes': num_classes,
                      'vocab size': vocab_size,
                      'Embedding Size': embedding_dim,
                      'head number': nhead,
                      'hidden size': d_hid,
                      'layers': nlayers,
                      'dropout': dropout}
            with st.spinner(f"#### Processing the data and label, please wait..."):
                # train_data, val_data = train_test_split(df[['first_name', 'gender_code']], test_size=0.2, random_state=42)
                # X
                train_data['encoded_names'] = train_data[first_name_option].apply(lambda name: encode_name(name))
                val_data['encoded_names'] = val_data[first_name_option].apply(lambda name: encode_name(name))  
                train_sequences = train_data['encoded_names'].tolist()
                val_sequences = val_data['encoded_names'].tolist()
                # y
                train_labels = train_data['gender_code'].values
                val_labels = val_data['gender_code'].values
                # max name length
                max_name_length = max(len(name) for name in train_sequences)
                with col_display_parameter:
                    st.write(f'max name length is {max_name_length}')
                for i in range(len(train_sequences)):
                    if len(train_sequences[i]) < max_name_length:
                        train_sequences[i] = train_sequences[i] + [0] * (max_name_length - len(train_sequences[i]))
                    elif len(train_sequences[i]) > max_name_length:
                        train_sequences[i] = train_sequences[i][:max_name_length]
                for i in range(len(val_sequences)):
                    if len(val_sequences[i]) < max_name_length:
                        val_sequences[i] = val_sequences[i] + [0] * (max_name_length - len(val_sequences[i]))
                    elif len(val_sequences[i]) > max_name_length:
                        val_sequences[i] = val_sequences[i][:max_name_length]
                train_sequences = torch.LongTensor(train_sequences)
                val_sequences = torch.LongTensor(val_sequences)
                train_labels = torch.LongTensor(train_labels)
                val_labels = torch.LongTensor(val_labels)
                train_dataset = TensorDataset(train_sequences, train_labels)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_dataset = TensorDataset(val_sequences, val_labels)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                with col_display_parameter:
                    model_parameter['max_name_length'] = max_name_length
                    st.write(model_parameter)
            with st.spinner(f"#### Training the NN model, please wait..."):
                start_time = time.time()
                model = TransformerModel(ntoken=vocab_size, d_model=embedding_dim, 
                                     nhead=nhead, d_hid=d_hid, nlayers=nlayers, dropout=dropout, 
                                     sequence_length=max_name_length)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()
                num_epochs = 1000
                best_val_loss = np.inf  # Set an initial high value for best validation loss
                patience = 30  # Number of epochs to wait for improvement
                wait = 0 
                col_tr, col_va, col_time = st.columns(3)
                for epoch in range(num_epochs):
                    start_time_ = time.time()
                    model.train()
                    total_loss = 0.0
                    for sequences, labels in train_loader:
                        optimizer.zero_grad()
                        output = model(sequences)
                        loss = criterion(output, labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    avg_loss = total_loss / len(train_loader)
                    if (epoch + 1) % 10 == 0:
                        with col_tr:
                            st.write(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for val_sequences, val_labels in val_loader:
                            val_output = model(val_sequences)
                            val_loss += criterion(val_output, val_labels).item()
                    avg_val_loss = val_loss / len(val_loader)
                    if (epoch + 1) % 10 == 0:
                        with col_va:
                            st.write(f"Validation Loss: {avg_val_loss:.4f}")
                        with col_time:
                            end_time_ = time.time()
                            st.write(f"10 Epoches time: {format_duration((end_time_ - start_time_)*10)}")
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        wait = 0  # Reset patience counter
                    else:
                        wait += 1  # Increment patience counter
                    if wait >= patience:
                        st.write("Early stopping triggered. No improvement in validation loss.")
                        break  # Stop training
                    
                end_time = time.time()
                st.success(f"Training spent: {format_duration(end_time - start_time)}")
            with st.spinner(f"#### Model Evaluation, please wait..."):
                st.write(f"#### Model Metric")
                # eval model result
                all_predictions = []
                all_true_labels = []
                model.eval()
                with torch.no_grad():
                    for val_sequences, val_labels in val_loader:
                        val_output = model(val_sequences)
                        val_probs = torch.softmax(val_output, dim=1)
                        # Get the predicted class (argmax)
                        val_preds = torch.argmax(val_probs, dim=1)
                        item_preds = [item for item in val_preds.tolist()]
                        all_predictions = all_predictions + item_preds
                        all_true_labels = all_true_labels + val_labels.tolist()
                # st.write('Wrong Predict Name:')
                # for idx in range(0, len(all_true_labels)):
                #     if all_true_labels[idx] != all_predictions[idx]:
                #         st.write(all_predictions[idx])
                #         st.write(val_data.iloc[idx])
                # all_predictions = [item for item in all_predictions]
                report = classification_report(all_true_labels, all_predictions, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                
                # report_df = pd.DataFrame(report).transpose()
                col_NN, col_voc = st.columns(2)
                with col_NN:
                    msg = log_mdoel(model_name = 'NN_Transformer',
                                        model = model, 
                                        result = report['weighted avg'],
                                        data_size = data_size,
                                        model_type = model_options,
                                        model_parameter = model_parameter)
                    # torch.save(model.state_dict(), "models/NN.pth")
                    st.success(msg)
                


        if model_options == 'Logistic':
            with st.spinner(f"Preparing data + Training..."):
                X_train, X_test, y_train, y_test = train_data[[first_name_option]], val_data[[first_name_option]], train_data[['gender_code']], val_data[['gender_code']]
                vectorizer = CountVectorizer()
                X_train_vectorized = vectorizer.fit_transform(X_train[first_name_option].values)
                X_test_vectorized = vectorizer.transform(X_test[first_name_option].values)
                # st.write(X_test_vectorized)

                classifierSearch = LogisticRegression(solver='liblinear', class_weight='balanced')
                param_grid = {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2']
                }
                grid_search = GridSearchCV(classifierSearch, param_grid, cv=10, scoring='f1_weighted')
                grid_search.fit(X_train_vectorized, y_train)
                st.write(grid_search.best_estimator_)
                classifier = grid_search.best_estimator_
                classifier.fit(X_train_vectorized, y_train)
                y_pred = classifier.predict(X_test_vectorized)
                # Evaluate the model
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                # log model
                col_logistic, col_countvectorizer = st.columns(2)
                with col_logistic:
                    if st.button('Log logistic_gender Model'):
                        msg = log_mdoel(model_name = ['logistic_gender.pkl', 'countvectorizer_gender.pkl'],
                                        model = [classifier,  vectorizer], 
                                        result = report['weighted avg'],
                                        data_size= data_size,
                                        model_type  = model_options,
                                        model_parameter = None)
                        # joblib.dump(classifier, 'models/logistic_gender.pkl')
                        # joblib.dump(vectorizer, 'models/countvectorizer_gender.pkl')
                        st.success(msg)



        if model_options == 'CatBoost':
            with st.spinner(f"Preparing data + Training..."):
                train_data_ = train_data.rename(columns={first_name_option : 'first_name'})
                val_data_ = val_data.rename(columns={first_name_option : 'first_name'})
                X_train, X_test, y_train, y_test = train_data_[['first_name']], val_data_[['first_name']], train_data[['gender_code']], val_data[['gender_code']]
                cat_features = ['first_name']
                train_data = Pool(data=X_train, label=y_train, cat_features=cat_features)
                test_data = Pool(data=X_test, cat_features=cat_features)
                catB_model = CatBoostClassifier(iterations=2000, 
                                                depth=6, 
                                                learning_rate=0.1, 
                                                loss_function='Logloss', 
                                                verbose=100)
                catB_model.fit(train_data)
                st.write(catB_model.best_score_)
                y_pred = catB_model.predict(test_data)
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                if st.button('Log CatBoost Model'):
                        msg = log_mdoel(model_name = 'catboost_model', 
                                        model = catB_model,  
                                        result = report['weighted avg'],
                                        data_size= data_size,
                                        model_type  = model_options,
                                        model_parameter = None)
                        # catB_model.save_model('models/catboost_model')
                        st.success(msg)





       



if __name__ == '__main__':
    app()
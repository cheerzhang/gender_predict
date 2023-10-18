import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, cross_val_predict
import matplotlib.pyplot as plt
import mlflow
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import catboost
from catboost import CatBoostClassifier, Pool
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim



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


class NameEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(NameEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8)
        self.fc = nn.Linear(embedding_dim * 26, output_dim)  # For classification


    def forward(self, x):
        embedded = self.embedding(x) # B, S, D
        embedded = embedded.permute(1, 0, 2) # S, B, D
        # Apply multi-head self-attention
        attention_output, _ = self.multihead_attention(embedded, embedded, embedded) # S, B , D
        attention_output = attention_output.permute(1, 0, 2)  # B, S , D
        B, S, D = attention_output.shape
        output = attention_output.reshape(B, -1)
        output = self.fc(output)
        # output = self.fc(embedded)
        return output

def encode_name(name):
    letter_to_number = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 
                        'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 
                        't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '.': 27}
    name = name.lower()  # Convert to lowercase for consistency
    encoded_name = [letter_to_number[letter] for letter in name if letter in letter_to_number]
    return encoded_name


def encode_label(label):
    encoded_label = [0, 0, 0]
    encoded_label[label] = 1
    return encoded_label





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
        options = st.selectbox('Chose Model', ('NN', 'CatBoost', 'Logistic+OVR',  'XGB'))
        if options == 'CatBoost':
            with st.spinner(f"traning CatBoost"):
                classifier = CatBoostClassifier(iterations=1000,  # Number of boosting iterations
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
        if options == 'NN':
            with st.spinner(f"traning Transformers"):
                train_data, val_data = train_test_split(df[['first_name', 'gender']], test_size=0.2, random_state=42)
                train_data['encoded_names'] = train_data['first_name'].apply(lambda name: encode_name(name))
                val_data['encoded_names'] = val_data['first_name'].apply(lambda name: encode_name(name))
                # train_data['encoded_labels'] = train_data['gender'].apply(lambda label: encode_label(label))
                # val_data['encoded_labels'] = val_data['gender'].apply(lambda label: encode_label(label))

                train_sequences = train_data['encoded_names'].tolist()
                val_sequences = val_data['encoded_names'].tolist()

                # train_labels = train_data['encoded_labels'].tolist()
                train_labels = train_data['gender'].values
                # val_labels = val_data['encoded_labels'].tolist()
                val_labels = val_data['gender'].values
                # st.dataframe(val_data)

                max_name_length = max(len(name) for name in train_sequences)
                st.write(f'max_name_length is {max_name_length}')
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
                batch_size = 32
                train_dataset = TensorDataset(train_sequences, train_labels)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_dataset = TensorDataset(val_sequences, val_labels)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

                num_classes = 3 # Assuming 26 letters in the alphabet
                input_dim =  28
                embedding_dim = 16
                model = NameEmbedding(input_dim, embedding_dim, num_classes)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                # optimizer = optim.SGD(model.parameters(), lr=0.001)

                criterion = nn.CrossEntropyLoss()
                num_epochs = 1000
                total_loss = 0.0 
                best_val_loss = np.inf  # Set an initial high value for best validation loss
                patience = 10  # Number of epochs to wait for improvement
                wait = 0 
                col_tr, col_va = st.columns(2)
                for epoch in range(num_epochs):
                    model.train()
                    for sequences, labels in train_loader:
                        optimizer.zero_grad()
                        output = model(sequences)
                        # st.write(output.shape)
                        # st.write(labels)
                        # labels_ = labels.view(-1, 1)
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
                            # val_labels_ = val_labels.view(-1, 1)
                            val_loss += criterion(val_output, val_labels).item()
                    avg_val_loss = val_loss / len(val_loader)
                    if (epoch + 1) % 10 == 0:
                        with col_va:
                            st.write(f"Validation Loss: {avg_val_loss:.4f}")
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        wait = 0  # Reset patience counter
                    else:
                        wait += 1  # Increment patience counter
                    if wait >= patience:
                        st.write("Early stopping triggered. No improvement in validation loss.")
                        break  # Stop training

                # eval model result
                all_predictions = []
                all_true_labels = []
                model.eval()
                with torch.no_grad():
                    for val_sequences, val_labels in val_loader:
                        val_output = model(val_sequences)
                        # val_labels_ = val_labels.view(-1, 1)
                        # Apply softmax to get class probabilities
                        val_probs = torch.softmax(val_output, dim=1)
                        # Get the predicted class (argmax)
                        val_preds = torch.argmax(val_probs, dim=1)
                        item_preds = [item for item in val_preds.tolist()]
                        # all_predictions.append(val_preds)
                        all_predictions = all_predictions + item_preds
                        # all_true_labels.append(val_labels)
                        # st.write(len(val_labels.tolist()))
                        all_true_labels = all_true_labels + val_labels.tolist()
                st.write(len(all_predictions))
                st.write(len(all_true_labels))
                report = classification_report(all_true_labels, all_predictions)
                st.dataframe(report)
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
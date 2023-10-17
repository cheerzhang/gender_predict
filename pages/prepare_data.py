import streamlit as st
import pandas as pd



@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# file download
def download_csv(df, file_name):
    csv = convert_df(df)
    st.download_button(
        label=f"Download {file_name} CSV",
        data=csv,
        file_name=f'[New]{file_name}.csv',
        mime='text/csv',
    )


def app():
    st.markdown('# Gender Data Preparing')
    df_exist = None
    col1, col2 = st.columns(2)
    with col2:
        df_file = st.file_uploader("Choose 'new gender' file :", key="gender_file_upload")
    with col1:
        df_exist_file = st.file_uploader("Choose 'exist gender' file :", key="gender_exist_file_upload")
    if df_file is not None:
        df_gender = pd.read_csv(df_file)
        with col2:
            st.write(f"new data size: :green[{df_gender.shape}]")
        first_name_column = st.selectbox('Select first name column', df_gender.columns.values, index=df_gender.columns.get_loc('first_name') if 'first_name' in df_gender.columns.values else 0)
        df_exist = pd.read_csv(df_exist_file)
        with col1:
            st.write(f"exist data size: :green[{df_exist.shape}]")
        # make all name lower
        df_exist['first_name'] = df_exist['first_name'].str.lower()
        df_gender[first_name_column] = df_gender[first_name_column].str.lower()
        # remvoe duplicate
        existing_first_names = set(df_exist['first_name'])
        df_gender = df_gender.rename(columns={first_name_column: 'first_name'})
        df_gender_filtered = df_gender[~df_gender['first_name'].isin(existing_first_names)]
        df_gender_filtered = df_gender_filtered[df_gender_filtered['first_name'].str.len() > 2]
        st.write(f"data to fill: {df_gender_filtered.shape}")
        df_gender_filtered['female'] = False
        df_gender_filtered['male'] = False
        df_gender_filtered['unknown'] = False

        col_label, col_table = st.columns([3,1])
        with col_label:
            df_label = st.data_editor(
                df_gender_filtered[['first_name', 'unknown', 'female', 'male']],
                column_config={
                    "unknown": st.column_config.CheckboxColumn(
                        "unknown?",
                        help="unknown?",
                        default=False,
                    ),
                    "female": st.column_config.CheckboxColumn(
                        "Female?",
                        help="Female?",
                        default=False,
                    ),
                    "male": st.column_config.CheckboxColumn(
                        "male?",
                        help="male?",
                        default=False,
                    )
                },
                disabled=['first_name'],
                hide_index=True,
                key = 'for_label'
            )
        with col_table:
            df_girl_ = df_label[df_label['female']==True]
            df_boy_ = df_label[df_label['male']==True]
            df_unknown_ = df_label[df_label['unknown']==True]
            df_unknown_['gender_code'] = 'U'
            df_girl_['gender_code'] = 'F'
            df_boy_['gender_code'] = 'M'
            result_df = pd.concat([df_girl_, df_boy_, df_unknown_], axis=0)
            result_df = result_df[['first_name', 'gender_code']]
        if df_exist is not None:
            df_exist_file_ = df_exist[['first_name', 'gender_code']]
            combine_df = pd.concat([result_df, df_exist_file_], axis=0)
            st.write(f"new file size: from:green[{df_exist_file_.shape}] to :green[{combine_df.shape}]")
            with col_table:
                st.dataframe(combine_df)
            download_csv(combine_df, 'train_gender')
        else:
            st.write(f"new file size: :green[{result_df.shape}]")
            with col_table:
                st.dataframe(result_df)
            download_csv(result_df, 'train_gender')


if __name__ == '__main__':
    app()
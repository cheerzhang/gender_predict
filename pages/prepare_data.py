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
    col1, col2 = st.columns(2)
    with col2:
        df_file = st.file_uploader("Choose 'new gender' file :", key="gender_file_upload")
        st.write(f"new data size: :green[{df_file.shape}]")
    with col1:
        df_exist_file = st.file_uploader("Choose 'exist gender' file :", key="gender_exist_file_upload")
        st.write(f"exist data size: :green[{df_exist_file.shape}]")
    if df_file is not None:
        df_gender = pd.read_csv(df_file)
        first_name_column = st.selectbox('Select first name column', df_gender.columns.values, index=df_gender.columns.get_loc('first_name') if 'first_name' in df_gender.columns.values else 0)
        df_exist = pd.read_csv(df_exist_file)
        # make all name lower
        df_exist['first_name'] = df_exist['first_name'].str.lower()
        df_gender[first_name_column] = df_gender[first_name_column].str.lower()
        # remvoe duplicate
        existing_first_names = set(df_exist['first_name'])
        df_gender = df_gender.rename(columns={first_name_column: 'first_name'})
        df_gender_filtered = df_gender[~df_gender['first_name'].isin(existing_first_names)]
        df_gender_filtered = df_gender_filtered[df_gender_filtered['first_name'].str.len() > 2]
        df_gender_filtered['gender'] = False
        df_gender_unknown = df_gender_filtered
        df_gender_for_girl = df_gender_filtered
        df_gender_for_boy = df_gender_filtered
        st.write(f"data to fill: {df_gender_filtered.shape}")

        # fill in the label
        col_unknown, col_f, col_m = st.columns(3)
        with col_unknown:
            st.markdown("Label for Unknown")
            df_unknown = st.data_editor(
                df_gender_unknown[['first_name', 'gender']],
                column_config={
                    "gender": st.column_config.CheckboxColumn(
                        "Unknown?",
                        help="Select gender",
                        default=False,
                    )
                },
                disabled=['first_name'],
                hide_index=True,
                key = 'for_unknown'
            )
            df_unknown_ = df_unknown[df_unknown['gender']==True]
            st.write(f"Unknowns have {df_unknown_.shape}")
        with col_f:
            st.markdown("Label for Female")
            df_girl = st.data_editor(
                df_gender_for_girl[['first_name', 'gender']],
                column_config={
                    "gender": st.column_config.CheckboxColumn(
                        "Female?",
                        help="Select gender",
                        default=False,
                    )
                },
                disabled=['first_name'],
                hide_index=True,
                key = 'for_girl'
            ) 
            df_girl_ = df_girl[df_girl['gender']==True]
            st.write(f"girls have {df_girl_.shape}")
        with col_m:
            st.markdown("Label for Male")
            df_boy = st.data_editor(
                df_gender_for_boy[['first_name', 'gender']],
                column_config={
                    "gender": st.column_config.CheckboxColumn(
                        "Male?",
                        help="Select gender",
                        default=False,
                    )
                },
                disabled=['first_name'],
                hide_index=True,
                key = 'for_boy'
            ) 
            df_boy_ = df_boy[df_boy['gender']==True]
            st.write(f"boys have {df_boy_.shape}")
        df_unknown_['gender_code'] = 'U'
        df_girl_['gender_code'] = 'F'
        df_boy_['gender_code'] = 'M'
        result_df = pd.concat([df_girl_, df_boy_, df_unknown_], axis=0)
        result_df = result_df[['first_name', 'gender_code']]
        st.dataframe(result_df)
        download_csv(result_df, 'train_gender')


if __name__ == '__main__':
    app()
import streamlit as st


st.markdown("# Gender predict model")


# Pages under Data Task
col1, col2 = st.columns(2)
with col1:
    with st.expander(" [[Data] Data prepare](/prepare_data)"):
        st.markdown("""
        This tool for preparing data including labeling for the data.
        """)
with col2:
    with st.expander(" [[Model] Train model](/prepare_data)"):
        st.markdown("""
        Train the gender predict model.
        """)
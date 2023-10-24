import streamlit as st


st.markdown("# Gender predict model")


# Pages under Data Task
col1, col2 = st.columns(2)
with col1:
    with st.expander(" [[Model] Train model](/model_train)"):
        st.markdown("""
        Train the gender predict model.
        """)
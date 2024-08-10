import streamlit as st

st.sidebar.title("RAG Experience")
page = st.sidebar.selectbox("Go to", ["PDF Uploader", "RAG Chat"])

# Import the pages
if page == "PDF Uploader":
    from src import pdf_frontend
    pdf_frontend.run()
elif page == "RAG Chat":
    from src import query_frontend
    query_frontend.run()

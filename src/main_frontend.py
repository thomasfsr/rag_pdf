import streamlit as st

st.sidebar.title("RAG Experience")
page = st.sidebar.selectbox("Go to", ["PDF Uploader", "RAG Chat"])

# Import the pages
if page == "PDF Uploader":
    import pdf_frontend
    pdf_frontend.run()
elif page == "RAG Chat":
    import simple_frontend
    simple_frontend.run()

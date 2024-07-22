import streamlit as st
import os
import glob
from dotenv import load_dotenv
load_dotenv()
from time import sleep

key = os.getenv('openai_key')

def run():
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0
    myform = st.form(key='Load PDF Files')
    with myform:
        uploaded_files = st.file_uploader("Insert PDFs you want to query", 
                                            type="pdf",
                                            accept_multiple_files=True,
                                            key=st.session_state["file_uploader_key"]
                                            )
        submit_button = st.form_submit_button('Upload PDF files')
        if submit_button:
            if not uploaded_files:
                st.error('No files to upload.')
            elif uploaded_files is not None:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("data", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success('Upload Concluded!')
                sleep(2)
                st.session_state["file_uploader_key"] += 1
                st.rerun()


    col1, col2, col3 = st.columns(3)
    with col1:
        button_show = st.button("Show saved PDF files :open_file_folder:")
        if button_show:
            saved_files = glob.glob("data/*.pdf")
            if saved_files:
                st.write("PDFs saved:")
                for saved_file in saved_files:
                    saved_file = os.path.basename(saved_file)
                    st.write(saved_file)
            else:
                st.write("No PDF files saved yet.")

    with col2:
        button_load = st.button(':green[Load PDF files to LanceDB] :inbox_tray:')
        if button_load:
            from vector_db import Embedding_Vector
            motor = Embedding_Vector(key, 'data/.lancedb','data')
            motor.add_to_lancedb(chunk_size=800, chunk_overlap=80)

    with col3:
        delete_button = st.button(':red[Delete PDF files] :wastebasket:')
        if delete_button:
            files = glob.glob("data/*.pdf")
            if files:
                for file_path in files:
                    os.remove(file_path)
                    file = os.path.basename(file_path)
                    st.write(f"Deleted: {file}")
                    pass
            else:
                st.write('No PDF files to delete.')
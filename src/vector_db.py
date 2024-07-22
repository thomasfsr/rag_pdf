import os
from dotenv import load_dotenv
# from langchain_community.llms import Ollama
# from langchain.chains import SimpleSequentialChain
# from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.lancedb import LanceDB
from lancedb import connect

load_dotenv()
key = os.getenv('openai-key')

class Embedding_Vector:

    def __init__(self, openai_key:str, path_db:str, dir_pdf_path:str=None):
        self.openai_key = openai_key
        self.path_db = path_db
        self.dir_pdf_path = dir_pdf_path

    def model_openai(self):
        model = ChatOpenAI(api_key=self.openai_key)
        return model

    def load_documents(self, path:str):
        doc_loader = PyPDFDirectoryLoader(path, glob= "*.pdf")
        return doc_loader.load()

    def split_documents(self, documents:list[Document], chunk_size:int, chunk_overlap:int):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function  = len,
            is_separator_regex=False
        )
        return text_splitter.split_documents(documents)


    def calculate_chunk_ids(self, chunks:list[Document]):
        last_page_id = None
        current_chunk_index = 0
        for chunk in chunks:
            source = chunk.metadata.get('source')
            page = chunk.metadata.get('page')
            current_page_id = f'{source}:{page}'
            if current_page_id == last_page_id:
                current_chunk_index +=1
            else:
                current_chunk_index = 0
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

        return chunks

    def get_embedding_function(self):
        embeddings = OpenAIEmbeddings(api_key=self.openai_key)
        return embeddings

    def add_to_lancedb(self, chunk_size:int, chunk_overlap:int):
        os.makedirs(self.dir_pdf_path, exist_ok=True)
        con = connect(self.path_db)
        db = LanceDB(connection=con, embedding=self.get_embedding_function())

        documents = self.load_documents(path = self.dir_pdf_path)
        chunks = self.split_documents(documents,chunk_size, chunk_overlap)

        chunks_with_ids = self.calculate_chunk_ids(chunks)
        # existing_items = db.get(include=[])  # IDs are always included by default
        table = db.get_table()
        if table is not None:
            existing_items = table.to_pandas()  # Convert table to pandas DataFrame
            existing_ids = set(existing_items["id"])  # Assuming the ID column is named "id"
        else:
            existing_ids = set()
        print(f"Number of existing documents in DB: {len(existing_ids)}")
        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)
        if len(new_chunks):
            print(f"Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids = new_chunk_ids)
        else:
            print("No new documents to add")


if __name__ == '__main__':
    motor = Embedding_Vector(key, 'data/.lancedb','data')
    motor.add_to_lancedb(chunk_size=800, chunk_overlap=80)
    # islp_pdf = 'data'
    # documents = load_documents(islp_pdf)
    # chunks = split_documents(documents)
    # add_to_lancedb(path_db='data/lancedb',chunks=chunks, openai_key=key)

# def generate_text(prompt: str):
#     response = llm.generate([prompt])
#     return response[0]['text']

# # Example usage
# prompt = "Tell me a story about a brave knight."
# response = generate_text(prompt)
# print(response)

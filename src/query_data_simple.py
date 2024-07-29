from langchain_community.vectorstores.lancedb import LanceDB
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from lancedb import connect

from src.vector_db import Embedding_Vector

from dotenv import load_dotenv
import os
import yaml

load_dotenv()
key = os.getenv('openai_key')

LANCE_PATH = "data/lancedb"

with open('src/prompt_template.yml', 'r') as pt:
    PROMPT_TEMPLATE = yaml.safe_load(pt)['prompt_template']
class LLM_Rag:

    def __init__(self, lance_path:str, openai_key:str, k:int):
        self.lance_path = lance_path
        self.openai_key = openai_key
        self.k = k
        self.llm = ChatOpenAI(api_key=self.openai_key, model="gpt-3.5-turbo",temperature=0.4)

        self.default_template = ChatPromptTemplate.from_messages(
            [
                ('system', 'you are a helpful assistant that responds on the language of the question given.'),
                ('human', '''Answer the question based ONLY on the following context: \n {context} \n\n --- \n\n
                 Answer the following question based on the above context: \n {question}''')
            ]
        )

    def query_rag(self, query_text: str):
        con = connect(self.lance_path)
        ev = Embedding_Vector(openai_key=self.openai_key, path_db='data/.lancedb')
        db = LanceDB(connection=con, embedding= ev.get_embedding_function())

        retriever = db.similarity_search_with_score(query_text, k=self.k)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in retriever])

        chain = self.default_template | self.llm | StrOutputParser()
        response_text = chain.invoke(input={'context':context_text,
                                            'question':query_text})

        sources = [(doc.metadata.get("id", None), score) for doc, score in retriever]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        return response_text, formatted_response

if __name__ == '__main__':
    llm = LLM_Rag(lance_path='data/.lancedb', openai_key=key, k=4)
    response, fr= llm.query_rag('Quais os doces finos?')
    print(response)
    # response, fr= llm.query_rag('Qual o preço unitário da trufa de maracujá?')
    # print(response)
    # response, fr= llm.query_rag('Quanto custa 10 shiny shells?')
    # print(response)
    # response, _ = llm.query_rag('Quanto fica 20 shiny shell, 20 tartelette belga, 20 piramide de whisky e 30 pavlova?')
    # print(response)
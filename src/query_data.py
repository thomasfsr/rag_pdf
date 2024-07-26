from langchain_community.vectorstores.lancedb import LanceDB
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from lancedb import connect

from vector_db import Embedding_Vector

from dotenv import load_dotenv
import os
import yaml

load_dotenv()
key = os.getenv('openai_key')

LANCE_PATH = "data/lancedb"

with open('src/prompt_template.yml', 'r') as pt:
    PROMPT_TEMPLATE = yaml.safe_load(pt)['prompt_template']
class LLM_Rag:

    def __init__(self, prompt_template:str, lance_path:str, openai_key:str, k:int):
        self.prompt_template = prompt_template
        self.lance_path = lance_path
        self.openai_key = openai_key
        self.k = k
        self.llm = ChatOpenAI(api_key=self.openai_key, model="gpt-3.5-turbo",temperature=0.4)

    def query_rag(self, query_text: str):
        con = connect(self.lance_path)
        ev = Embedding_Vector(openai_key=self.openai_key, path_db='data/.lancedb')
        db = LanceDB(connection=con, embedding= ev.get_embedding_function())

        results = db.similarity_search_with_score(query_text, k=self.k)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = self.llm
        response_text = model.invoke(prompt)

        sources = [(doc.metadata.get("id", None), score) for doc, score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        return response_text, formatted_response

if __name__ == '__main__':
    llm = LLM_Rag(prompt_template=PROMPT_TEMPLATE, lance_path='data/.lancedb', openai_key=key, k=6)
    response, fr= llm.query_rag('Quais os doces finos?')
    print(response.content)
    response, fr= llm.query_rag('Qual o preço unitário da trufa de maracujá?')
    print(response.content)
    response, fr= llm.query_rag('Quanto custa 10 shiny shells?')
    print(response.content)
    response, _ = llm.query_rag('Quanto fica 20 shiny shell, 20 tartelette belga, 20 piramide de whisky e 30 pavlova?')
    print(response.content)
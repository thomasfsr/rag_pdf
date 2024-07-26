from langchain_community.vectorstores.lancedb import LanceDB
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain.chains.llm_math.base import LLMMathChain
from lancedb import connect

from vector_db import Embedding_Vector
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv('openai_key')

LANCE_PATH = "data/lancedb"

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

        self.classify_template = ChatPromptTemplate.from_messages(
            [
                ('system', 'you are a helpful assistant.'),
                ('human', '''Classify this question as yes if it requires calculation like 
                 multiplication of values and sum of quantities and prices, otherwise no: \n {question}''')
            ]
        )

        self.tool_template = ChatPromptTemplate.from_messages([('system',"""Answer the following questions as best you can. You have access to the following tools: \n
                {tools} \n
                You have to consult the following context: {context}
                Use the following format: \n
                Question: the input question you must answer with the context \n
                Thought: you should always think about what to do \n
                Action: the action to take, should be one of [{tool_names}] \n
                Action Input: the input to the action \n
                Observation: the result of the action \n
                ... (this Thought/Action/Action Input/Observation can repeat N times) \n
                Thought: I now know the final answer \n
                Final Answer: the final answer to the original input question \n
                Begin! \n
                Question: {question} \n
                Context: {context} \n
                Thought:{agent_scratchpad}""")])
        
        self.tool = [Tool(name="Calculator",
                           func=LLMMathChain.from_llm(llm=self.llm).run,
                           description='''Para fazer calculos.''')]

    def query_rag(self, query_text: str):
        con = connect(self.lance_path)
        ev = Embedding_Vector(openai_key=self.openai_key, path_db='data/.lancedb')
        db = LanceDB(connection=con, embedding= ev.get_embedding_function())

        retriever = db.similarity_search_with_score(query_text, k=self.k)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in retriever])

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tool,
            prompt=self.tool_template,
        )
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=self.tool, handle_parsing_errors=True, verbose=True
        )

        branches = RunnableBranch(
            (lambda x: 'yes' in x.lower(),
             agent_executor),
            (lambda x: 'no' in x.lower(),
             self.default_template | self.llm | StrOutputParser()),
             self.default_template | self.llm | StrOutputParser()
             )


        chain_class = self.classify_template | self.llm | StrOutputParser()
        chain = chain_class | branches
        response_text = chain.invoke(input={'context':context_text,
                                            'question':query_text})

        # sources = [(doc.metadata.get("id", None), score) for doc, score in retriever]
        # formatted_response = f"Response: {response_text}\nSources: {sources}"
        return response_text
if __name__ == '__main__':
    llm = LLM_Rag(lance_path='data/.lancedb', openai_key=key, k=4)
    response= llm.query_rag('Quais os doces finos?')
    print(response)
    response= llm.query_rag('Qual o preço unitário da trufa de maracujá?')
    print(response)
    response= llm.query_rag('Quanto custa 10 shiny shells?')
    print(response)
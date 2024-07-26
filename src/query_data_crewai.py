from langchain_community.vectorstores.lancedb import LanceDB

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import  create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains.llm_math.base import LLMMathChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from langchain.agents import initialize_agent, create_json_chat_agent, create_react_agent, Tool, AgentExecutor

from langchain.tools import BaseTool, StructuredTool, tool

from langchain_core.messages import AIMessage, HumanMessage

from langchain.agents.agent_types import AgentType

from langchain_openai.chat_models import ChatOpenAI

from langchain import hub

from crewai import Agent, Task, Crew, Process

from lancedb import connect

import json

from dotenv import load_dotenv
import os
import yaml

try:
    from src.vector_db import Embedding_Vector
except:
    from src.vector_db import Embedding_Vector

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
        self.chat_history = []  
        self.llm = ChatOpenAI(api_key=self.openai_key, model="gpt-3.5-turbo",temperature=0.4)

    
    def run(self):

        con = connect(self.lance_path)
        ev = Embedding_Vector(openai_key=self.openai_key, path_db='data/.lancedb')
        db = LanceDB(connection=con, embedding=ev.get_embedding_function())
        retriever= db.as_retriever()

        contextualize_q_system_prompt = (
            # """Dado um histórico de chat e a última pergunta do usuário, 
            # que pode referenciar o contexto no histórico do chat, 
            # formule uma pergunta independente que possa ser entendida sem o histórico do chat. 
            # NÃO responda à pergunta, apenas reformule-a, se necessário, e, caso contrário, 
            # retorne-a como está."""
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is.")

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt)
        
        qa_system_prompt = (
            # """Você é um assistente para tarefas de perguntas e respostas. 
            # Use os seguintes trechos de contexto recuperado para responder à pergunta. 
            # Se você não souber a resposta, apenas diga que não sabe.
            # Responda em português se a pergunta estiver em português.
            # """"
            "You are an assistant for question-answering tasks. Use "
            "the following pieces of retrieved context to answer the "
            "question. If you don't know the answer, just say that you know. "
            "\n\n"
            "{context}")
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain)


        # Set Up ReAct Agent with Document Store Retriever
        # Load the ReAct Docstore Prompt
        react_docstore_prompt = hub.pull("hwchase17/react")

        tools = [
        Tool(
            name="Responder Perguntas",
            func=lambda input, **kwargs: rag_chain.invoke(
                {"input": input, "chat_history": kwargs.get("chat_history", [])}
            ),
            description="To Answer based on the context.",
                ),
        Tool(name="Calculator",
                           func=LLMMathChain.from_llm(llm=self.llm).run,
                           description='''Para fazer calculos.''')
                ]
        
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=react_docstore_prompt,
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, handle_parsing_errors=True, verbose=True
        )

        chat_history = []
        while True:
            query = input("You: ")
            if query.lower() == "exit":
                break
            response = agent_executor.invoke(
                {"input": query, "chat_history": chat_history})
            print(f"AI: {response['output']}")

            # Update history
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=response["output"]))


if __name__ == '__main__':
    # llm = LLM_Rag(prompt_template=PROMPT_TEMPLATE, lance_path='data/.lancedb', openai_key=key, k=4)
    # response= llm.query_rag('Quais sabores de tartelette existem?')
    # print(response)
    model = LLM_Rag(prompt_template=PROMPT_TEMPLATE, lance_path='data/.lancedb', openai_key=key, k=4)
    result = model.run()

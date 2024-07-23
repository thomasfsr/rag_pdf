from langchain_community.vectorstores.lancedb import LanceDB

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import  create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains.llm_math.base import LLMMathChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from langchain.agents import initialize_agent, create_json_chat_agent, create_react_agent, Tool, AgentExecutor

from langchain.tools import BaseTool, StructuredTool, tool

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
    from vector_db import Embedding_Vector

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
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )

        # Create a prompt template for contextualizing questions
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create a history-aware retriever
        # This uses the LLM to help reformulate the question based on chat history
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # Answer question prompt
        # This system prompt helps the AI understand that it should provide concise answers
        # based on the retrieved context and indicates what to do if the answer is unknown
        qa_system_prompt = (
            "Responde based only on the context given and the historical chat."
            "\n\n"
            "{context}"
        )

        # Create a prompt template for answering questions
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create a chain to combine documents for question answering
        # `create_stuff_documents_chain` feeds all retrieved context into the LLM
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        # Create a retrieval chain that combines the history-aware retriever and the question answering chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


        # Function to simulate a continual chat
        # def continual_chat():
        print("Start chatting with the AI! Type 'exit' to end the conversation.")
        chat_history = []  # Collect chat history here (a sequence of messages)
        while True:
            query = input("You: ")
            if query.lower() == "exit":
                break
            # Process the user's query through the retrieval chain
            result = rag_chain.invoke({"input": query, "chat_history": chat_history})
            # Display the AI's response
            print(f"AI: {result['answer']}")
            # Update the chat history
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=result["answer"]))


if __name__ == '__main__':
    # llm = LLM_Rag(prompt_template=PROMPT_TEMPLATE, lance_path='data/.lancedb', openai_key=key, k=4)
    # response= llm.query_rag('Quais sabores de tartelette existem?')
    # print(response)
    model = LLM_Rag(prompt_template=PROMPT_TEMPLATE, lance_path='data/.lancedb', openai_key=key, k=4)
    result = model.run()

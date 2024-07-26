from langchain_community.vectorstores.lancedb import LanceDB
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain.chains.llm_math.base import LLMMathChain
from lancedb import connect

from src.vector_db import Embedding_Vector

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
        self.llm = ChatOpenAI(api_key=self.openai_key, model="gpt-3.5-turbo",temperature=0)

        self.default_template = ChatPromptTemplate.from_messages(
            [
                ('system', 'you are a helpful assistant.'),
                ('human', '''Answer the question based ONLY on the following context: \n {context} \n\n --- \n\n
                 Answer politely the following question based on the above context: \n {question}''')
            ]
        )

        self.final_template = ChatPromptTemplate.from_messages(
            [
                ('system', 'you are a helpful assistant that translates the final answer to the question language.'),
                ('human', '''You were given the following context: \n {context} \n\n --- \n\n
                 Answer the following question based on the above context explaining everything in details: \n {question}''')
            ]
        )

        self.translate_template = ChatPromptTemplate.from_messages(
            [
                ('system', 'you are a translator.'),
                ('human', '''ONLY identifies the language in this question to be used on the text. Do NOT return it in the result: \n {question}\n\n --- \n\n
                 Translate the following text to the language identified, MUST maintain its structure intact: \n {answer}
                 ''')
            ]
        )

        self.classify_template = ChatPromptTemplate.from_messages(
            [
                ('system', 'you are a helpful assistant.'),
                ('human', '''You should have to classify as YES if the
                 question given is asking for more than one product and their prices to calculate for some quantities.
                 If YES, next to YES add > and a list of the product, quantity, - unit price with UN. Extract ONLY from the context.
                 , otherwise NO. \n---\n 
                 Question: \n {question} \n---\n
                 Context: {context}''')
            ]
        )

        self.math_tool_template = ChatPromptTemplate.from_messages([('system',"""Answer the following questions as best you can. You have access to the following tools: \n
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
                Final Answer: the final answer to the original input question, Show the steps. \n
                Begin! \n
                Question: {question} \n
                Context: {context} \n
                Thought:{agent_scratchpad}""")])
        
        self.math_tool = [Tool(name="Calculator",
                           func=LLMMathChain.from_llm(llm=self.llm).run,
                           description='''To calculate product and prices. Always show the calculation made.''')]

    def query_rag(self, query_text: str):
        con = connect(self.lance_path)
        ev = Embedding_Vector(openai_key=self.openai_key, path_db='data/.lancedb')
        db = LanceDB(connection=con, embedding= ev.get_embedding_function())

        results = db.similarity_search_with_score(query_text, k=self.k)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        math_agent = create_react_agent(
            llm=self.llm,
            tools=self.math_tool,
            prompt=self.math_tool_template,
        )
        math_agent_executor = AgentExecutor.from_agent_and_tools(
            agent=math_agent, tools=self.math_tool, handle_parsing_errors=True, verbose=False
        )

        model = self.llm
        chain = self.classify_template | model | StrOutputParser()
        class_response = chain.invoke(input={'context': context_text, 'question': query_text})
        if class_response.lower() == 'no':
            chain = self.default_template | model | StrOutputParser()
            response_text = chain.invoke(input={'context': context_text, 'question': query_text})
        elif class_response.split(' >')[0].lower() == 'yes':
            list_products = class_response.split('>')[1]
            context_text = context_text + list_products
            math_response_text = math_agent_executor.invoke({'context': context_text, 'question': query_text})['output']
            final_chain = self.final_template | model | StrOutputParser()
            final_context = list_products + math_response_text
            final_response_text =final_chain.invoke({'context': final_context, 'question': query_text})
            translate_chain = self.translate_template | model | StrOutputParser()
            response_text = translate_chain.invoke({'answer': final_response_text, 'question': query_text})

        return response_text

if __name__ == '__main__':
    llm = LLM_Rag(lance_path='data/.lancedb', openai_key=key, k=6)
    # response = llm.query_rag('O que tem no Pedra da Lua?')
    # print(response)
    # response = llm.query_rag('Qual o preço unitário da trufa de maracujá?')
    # print(response)
    # response = llm.query_rag('Quanto custa 10 shiny shells?')
    # print(response)
    response = llm.query_rag('Quanto fica 20 shiny shell, 20 tartelette belga, 20 piramide de whisky e 30 pavlova?')
    print(response)
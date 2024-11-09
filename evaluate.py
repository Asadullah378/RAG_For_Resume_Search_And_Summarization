import json
from dotenv import load_dotenv
from prompt import AGENT_PROMPT_OPENAI
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.evaluation import load_evaluator


load_dotenv()

VECTOR_DB_FOLDER = "Vector-DB"

def load_existing_DB(path, embedding):
    # Load vector database
    db = Chroma(persist_directory=path,
                embedding_function=embedding)

    if db is None:
        raise Exception("Vector database not found")
    db.persist()
    return db

def initializeAgent():

    # Set Embeddings
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    # Set LLM
    kwargs = {"temperature": 0, "response_format": {  "type": "json_object" }}
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", **kwargs)

    evaluator = load_evaluator("trajectory", llm=llm)
    vectorDB = load_existing_DB(VECTOR_DB_FOLDER,embedding)

    # Create Retriever Tool
    retrieval_tool = create_retriever_tool(
        retriever=vectorDB.as_retriever(),
        name="Document_Retriever",
        description="Searches and returns the resumes that matches user requirements",
    )

    # Create Tool List (Currently we will only use retrieval tool)
    tools = [retrieval_tool]

    # Create Memory
    memory_key = "history"
    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm, max_token_limit=3000)

    # Set System Message
    system_message = SystemMessage(
        content=(AGENT_PROMPT_OPENAI)
    )

    # Create Prompt
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
    )

    # Create Agent
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

    # Create Agent Executer
    agent_executer = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    return agent_executer, evaluator


def getAnswer(agent_executer, evaluator, prompt):
    result = agent_executer({"input": prompt})
    evaluation_result = evaluator.evaluate_agent_trajectory(
        prediction=result["output"],
        input=result["input"],
        agent_trajectory=result["intermediate_steps"],
    )
    return evaluation_result

agent_executer, evaluator = initializeAgent()

while True:

    prompt = input("Enter Query: ")
    results = getAnswer(agent_executer, evaluator, prompt)
    print(results)

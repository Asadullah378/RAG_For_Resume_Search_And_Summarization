import json
import os
from dotenv import load_dotenv
from prompt import AGENT_PROMPT_OPENAI, AGENT_PROMPT_CLAUDE
from flask import Flask, request, jsonify, render_template
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_xml_agent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import boto3
from langchain_aws import ChatBedrock

load_dotenv()
app = Flask(__name__, static_folder='Dataset/Resumes', static_url_path='/resumes')

VECTOR_DB_FOLDER = "Vector-DB"
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

bedrock = boto3.client(
  service_name='bedrock-runtime',
  region_name='us-west-2',
  aws_access_key_id = aws_access_key_id,
  aws_secret_access_key = aws_secret_access_key, 
)

def load_existing_DB(path, embedding):
    # Load vector database
    db = Chroma(persist_directory=path,
                embedding_function=embedding)

    if db is None:
        raise Exception("Vector database not found")
    db.persist()
    return db

def initializeAgent_openai(model_name):

    # Set Embeddings
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    # Set LLM
    kwargs = {"temperature": 0, "response_format": {  "type": "json_object" }}
    llm = ChatOpenAI(model_name=model_name, **kwargs)

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

    return agent_executer

def initializeAgent_claude(modelId):

    # Set Embeddings
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    model_kwargs =  { 
    "max_tokens": 4096,
    "temperature": 0.0,
    }

    # Set LLM
    llm = ChatBedrock(
        client=bedrock,
        model_id=modelId,
        model_kwargs=model_kwargs)

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

    memory = ConversationBufferWindowMemory(
                memory_key=memory_key,
                k=5,
                return_messages=True,
                input_key="input",
                output_key="output",
            )
    # # Set System Message
    # system_message = SystemMessage(
    #     content=(AGENT_PROMPT)
    # )

    # # Create Prompt
    prompt = PromptTemplate.from_template(template=AGENT_PROMPT_CLAUDE)

    # Create Agent
    agent = create_xml_agent(llm=llm, tools=tools, prompt=prompt)

    # Create Agent Executer
    agent_executer = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    return agent_executer


def getAnswer(agent_executer, prompt):
    result = agent_executer({"input": prompt})
    answer = result["output"]
    answer = json.loads(answer)
    return answer


agent_executer_gpt_3_5 = initializeAgent_openai('gpt-3.5-turbo')
agent_executer_gpt_4 = initializeAgent_openai('gpt-4-turbo')
agent_executer_haiku = initializeAgent_claude('anthropic.claude-3-haiku-20240307-v1:0')
agent_executer_sonnet = initializeAgent_claude('anthropic.claude-3-sonnet-20240229-v1:0')
agent_executer_opus = initializeAgent_claude('anthropic.claude-3-opus-20240229-v1:0')

models = {
    'gpt-3.5' : agent_executer_gpt_3_5,
    'gpt-4' : agent_executer_gpt_4,
    'haiku' : agent_executer_haiku,
    'sonnet' : agent_executer_sonnet,
    'opus' : agent_executer_opus,
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_resumes():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400
    
    prompt = data['prompt']
    model_id = data['model']

    try:
        response = getAnswer(models[model_id], prompt)
        return jsonify(response)
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)

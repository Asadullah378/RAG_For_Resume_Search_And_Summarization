import json
from dotenv import load_dotenv
from prompt import AGENT_PROMPT_CLAUDE
from flask import Flask, request, jsonify, render_template
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_xml_agent
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain import PromptTemplate


load_dotenv()
app = Flask(__name__, static_folder='Dataset/Resumes', static_url_path='/resumes')

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
    llm = ChatAnthropic(model='claude-3-haiku-20240307', max_tokens=4096)

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


agent_executer = initializeAgent()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_resumes():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400
    
    prompt = data['prompt']
    try:
        response = getAnswer(agent_executer, prompt)
        return jsonify(response)
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

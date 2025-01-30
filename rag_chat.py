import json
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.cosmos import CosmosClient
import os
from dotenv import load_dotenv
from typing import List
from tenacity import retry, stop_after_attempt, wait_random_exponential 
import logging
from openai import AzureOpenAI
import asyncio
import panel as pn
import mlflow, mlflow.pyfunc
import time

# Load environment variables from .env file
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_ORCH_MODEL = os.getenv("AZURE_ORCH_MODEL")
COSMOS_URL = os.getenv("COSMOS_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")
AZURE_EMBEDDING = os.getenv("AZURE_EMBEDDING")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
DIMENSIONS = 1536

# Authenticate using Managed Identity (DefaultAzureCredential)
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

# Connect to CosmosDB
client = CosmosClient(COSMOS_URL, credential)
db = client.create_database_if_not_exists(DATABASE_NAME)

# Connect to OpenAI
openai_client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY, api_version="2023-05-15")


# MLflow Tracking
# Set MLflow to track experiments locally
mlflow_tracking_uri = "http://127.0.0.1:5000"  # Default MLflow UI URL
mlflow.set_tracking_uri(mlflow_tracking_uri)
logging.getLogger("mlflow").setLevel(logging.ERROR)
# Ensure async logging is still enabled
mlflow.config.enable_async_logging()

# Set the experiment
mlflow.set_experiment("Autogen Multi-Agent RAG Tracing")
###############
# Vector Search
###############

@retry(wait=wait_random_exponential(min=2, max=300), stop=stop_after_attempt(20))
def get_vector_embedding(text):
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=AZURE_EMBEDDING,
            dimensions=DIMENSIONS
        )
        return response.model_dump()['data'][0]['embedding']
    except Exception as e:
        logging.error("Error generating embeddings.", exc_info=True)
        raise

def vector_search(container, vectors, similarity_score=0.02, num_results=5):
    """
    Executes a vector similarity search query in CosmosDB.
    """
    query = '''
    SELECT TOP @num_results c, VectorDistance(c.embedding, @embedding) as SimilarityScore 
    FROM c
    WHERE VectorDistance(c.embedding, @embedding) > @similarity_score
    ORDER BY VectorDistance(c.embedding, @embedding)
    '''

    results = container.query_items(
        query=query,
        parameters=[
            {"name": "@embedding", "value": vectors},
            {"name": "@num_results", "value": num_results},
            {"name": "@similarity_score", "value": similarity_score}
        ],
        enable_cross_partition_query=True
    )
    
    formatted_results = []
    for result in results:
        if 'c' in result:
            document = result['c']
            document.pop('embedding', None)  # Remove embedding field
            formatted_results.append({
                'document': document,
                'similarity_score': result['SimilarityScore']
            })

    return formatted_results

def retrieve_results(query: str, container_names: List[str], num_results=5) -> List[dict]:
    """
    Retrieve vector search results from multiple CosmosDB containers.
    
    Args:
        query (str): The query to generate embeddings.
        container_names (List[str]): List of container names to search in.
        num_results (int): Number of top results to return.
    
    Returns:
        List[dict]: Aggregated and sorted search results.
    """
    embedding = get_vector_embedding(query)
    all_results = []

    # Iterate through each container and perform vector search
    for container_name in container_names:
        container = db.get_container_client(container_name)
        results = vector_search(container, embedding, num_results=num_results)
        all_results.extend(results)  # Append results from this container

    # Sort all results by similarity score (lower is more similar)
    all_results.sort(key=lambda x: x['similarity_score'])
    print(all_results)
    # Return top `num_results` from all containers combined
    return all_results

###################
# Chat Completion #
###################

def generate_completion(user_prompt, vector_search_results, chat_history):
    """
    Generates AI completion using OpenAI with multi-turn context.
    """
    system_prompt = '''
    You are an intelligent assistant. Use previous chat history and retrieved documents to answer questions.
    Do not use your own knowledge to answer questions. Only use chat history and retrieved documents.
    If you do not have enough information to answer a question, say so.
    '''

    messages = [{'role': 'system', 'content': system_prompt}]

    # Include previous chat turns
    for chat in chat_history:
        messages.append({'role': 'user', 'content': chat['prompt']})
        messages.append({'role': 'assistant', 'content': chat['completion']})

    # Current user question
    messages.append({'role': 'user', 'content': user_prompt})

    # Add vector search results as context
    for result in vector_search_results:
        messages.append({'role': 'system', 'content': json.dumps(result['document'])})

    # Get AI response
    response = openai_client.chat.completions.create(
        model=AZURE_ORCH_MODEL,
        messages=messages,
        temperature=0.1
    )    
    return response.model_dump()

#########
# Panel UI
#########

chat_feed = pn.Column(sizing_mode="stretch_width", height=500, scroll=True)
user_input = pn.widgets.TextInput(placeholder="Type your question...")
run_button = pn.widgets.Button(name="Ask RAG", button_type="primary")

agent_styles = {
    "user": {"background-color": "#D3E3FC", "padding": "8px", "border-radius": "8px"},
    "assistant_agent": {"background-color": "#C8E6C9", "padding": "8px", "border-radius": "8px"}
}

def add_message(sender, text):
    chat_feed.append(pn.pane.Markdown(f"**{sender}**: {text}", styles=agent_styles.get(sender, {})))

# Maintain chat history
chat_history = []  # List to store previous turns

async def run_chat():
    global chat_history  # Ensure we use global chat history

    task = user_input.value.strip()
    if not task:
        return

    add_message("user", task)
    user_input.value = ""  
    start_time = time.time()

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=f"Chat_{time.strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_param("User Query", task)
        mlflow.log_param("Start Time", time.strftime('%Y-%m-%d %H:%M:%S'))

        # Define the list of containers for vector search
        container_list = ["Complaints", "Fruits"]
        
        retrieved_data = retrieve_results(task, container_list, num_results=5)

        # Generate response using chat history
        response = generate_completion(task, retrieved_data, chat_history)

        # Extract response text
        response_text = response['choices'][0]['message']['content']

        # Add message to UI and history
        add_message("assistant_agent", response_text)

        # Update chat history for multi-turn
        chat_history.append({"prompt": task, "completion": response_text})

        mlflow.end_run()

def on_click(event):
    asyncio.create_task(run_chat())

run_button.on_click(on_click)

dashboard = pn.Column(
    pn.pane.Markdown("## 🤖 RAG Chat"),
    chat_feed,
    pn.Row(user_input, run_button)
)

dashboard.servable()

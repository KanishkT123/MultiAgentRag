from autogen_agentchat.agents import AssistantAgent
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.cosmos import CosmosClient
import os
from dotenv import load_dotenv
from typing import List
from tenacity import retry, stop_after_attempt, wait_random_exponential 
import logging
from openai import AzureOpenAI
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.base import TaskResult
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
AZURE_EMBEDDING= os.getenv("AZURE_EMBEDDING")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
DIMENSIONS = 1536

# Authenticate using Managed Identity (DefaultAzureCredential)
credential = DefaultAzureCredential()

# Create the token provider
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
# Connect to Azure Chat Model (for all agents)
az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=AZURE_ORCH_MODEL,
    model=AZURE_ORCH_MODEL,
    api_version="2024-08-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_ad_token_provider=token_provider
)

# Connect to CosmosDB
client = CosmosClient(COSMOS_URL, credential)
db = client.create_database_if_not_exists(DATABASE_NAME)

# Connect to OpenAI (for embedding)
openai_client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY, api_version="2023-05-15")

##########
# MLFlow #
##########
# MLflow Tracking
# Set MLflow to track experiments locally
mlflow_tracking_uri = "http://127.0.0.1:5000"  # Default MLflow UI URL
mlflow.set_tracking_uri(mlflow_tracking_uri)
logging.getLogger("mlflow").setLevel(logging.ERROR)
# Ensure async logging is still enabled
mlflow.config.enable_async_logging()

logging.getLogger("mlflow").setLevel(logging.ERROR)

experiment_name = "Autogen Multi-Agent RAG Tracing"
mlflow.set_experiment(experiment_name)

#########
# Tools #
#########

@retry(wait=wait_random_exponential(min=2, max=300), stop=stop_after_attempt(20))
def get_vector_embedding(text):
    try:        
        response = openai_client.embeddings.create(
            input=text,
            model=AZURE_EMBEDDING,
            dimensions=DIMENSIONS
        )
        embeddings = response.model_dump()
        return embeddings['data'][0]['embedding']
    except Exception as e:
        # Log the exception with traceback for easier debugging
        logging.error("An error occurred while generating embeddings.", exc_info=True)
        raise

def vector_search(container, vectors, fields, similarity_score=0.02, num_results=5):
    """
    Executes a vector similarity search query in CosmosDB.
    
    Args:
        container: CosmosDB container instance.
        vectors: The embedding vector for similarity comparison.
        fields: List of fields to retrieve from the documents.
        similarity_score: Threshold for similarity score filtering.
        num_results: Maximum number of results to return.
    
    Returns:
        List of documents matching the vector search criteria.
    """
    # Execute the query

    # Ensure fields is a list; join fields into a comma-separated string
    if not isinstance(fields, list):
        raise ValueError("Fields must be a list of column names.")

    selected_fields = ", ".join([f"c.{field}" for field in fields])  # Format field names

    # Construct the SQL query dynamically
    query = f'''
    SELECT TOP @num_results {selected_fields}, VectorDistance(c.embedding, @embedding) as SimilarityScore 
    FROM c
    WHERE VectorDistance(c.embedding, @embedding) > @similarity_score
    ORDER BY VectorDistance(c.embedding, @embedding)
    '''

    results = container.query_items(
        query= query,
        parameters=[
            {"name": "@embedding", "value": vectors},
            {"name": "@num_results", "value": num_results},
            {"name": "@similarity_score", "value": similarity_score}
        ],
        enable_cross_partition_query=True, populate_query_metrics=True)
    results = list(results)
    # Extract the necessary information from the results
    formatted_results = []
    for result in results:
        formatted_result = {
            'document': result
        }
        formatted_results.append(formatted_result)

    return formatted_results

def retrieve_results(query:str, containerName:str, fields:List[str]) -> List[str]:
    '''
    Retrieve results from the specified container using vector search.
    Args:
        query (str): The query to do the vector search for.
        containerName (str): The name of the container to search in - Either 'Complaints' or 'Fruits'.
        fields (str): The fields to retrieve from the documents
    '''
    container = db.get_container_client(containerName)
    embedding = get_vector_embedding(query)
    results = vector_search(container, embedding, fields)
    return results

def execute_sql(containerName: str, query: str, params: dict = None):
    '''
    Securely execute a SQL query using parameters in CosmosDB.
    
    Args:
        containerName (str): The name of the container to query.
        query (str): The parameterized SQL query.
        params (dict): The query parameters as a dictionary.
    
    Returns:
        List of results or an error message.
    '''
    container = db.get_container_client(containerName)

    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only read operations (SELECT queries) are allowed."

    # Securely execute the query with parameters
    results = list(container.query_items(
        query=query,
        parameters=[{"name": key, "value": value} for key, value in (params or {}).items()],
        enable_cross_partition_query=True
    ))
    
    return results if results else "No records found."

# Async function to retrieve input from UI
async def get_user_input(prompt: str, cancellation_token):
    while not user_input.value.strip():
        await asyncio.sleep(0.1)  # Wait for input asynchronously
    text = user_input.value.strip()
    user_input.value = ""  # Clear input field after reading
    return text

# Load system prompts from files
def load_prompt(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as file:
        return file.read()

##########
# Agents #
##########

assistant_agent = AssistantAgent(
    name = "assistant_agent",
    description = "An assistant agent that provides general information and assistance.",
    system_message = load_prompt("system_prompts/assistant_prompt.txt"),
    model_client=az_model_client,
    tools=[execute_sql, retrieve_results],
    reflect_on_tool_use=True
)

user_proxy = UserProxyAgent("user", input_func=get_user_input)

########
# Team #
########

# Create a group chat with the orchestrator and agents
text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=5)
termination = text_mention_termination

team = SelectorGroupChat(
    [assistant_agent, user_proxy],
    model_client=az_model_client,
    allow_repeated_speaker=True,
    termination_condition=termination,
)

#########
# Panel #
#########

# Create a chat display area
chat_feed = pn.Column(sizing_mode="stretch_width", height=500, scroll=True)

# User input field
user_input = pn.widgets.TextInput(placeholder="Type your question...")

# Run button
run_button = pn.widgets.Button(name="Ask Agents", button_type="primary")

# Agent styling dictionary (Converted to valid dict format)
agent_styles = {
    "user": {"background-color": "#D3E3FC", "padding": "8px", "border-radius": "8px", "margin": "5px", "width": "fit-content"},
    "assistant_agent": {"background-color": "#C8E6C9", "padding": "8px", "border-radius": "8px", "margin": "5px", "width": "fit-content", "align-self": "flex-end"},
}

# Function to append messages to chat
def add_message(sender, text):
    chat_feed.append(
        pn.pane.Markdown(
            f"**{sender}**: {text}",
            styles=agent_styles.get(sender, {"background-color": "#F1F1F1", "padding": "8px", "border-radius": "8px", "margin": "5px", "width": "fit-content"})
        )
    )

async def run_chat():
    task = user_input.value.strip()
    if not task:
        return

    add_message("user", task)  # Show user message
    user_input.value = ""  # Clear input field
    start_time = time.time()

    # Ensure any previous MLflow run is ended
    if mlflow.active_run():
        mlflow.end_run()

    # Start MLflow run for the chat session
    with mlflow.start_run(run_name=f"Chat_{time.strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_param("User Query", task)
        mlflow.log_param("Start Time", time.strftime('%Y-%m-%d %H:%M:%S'))

        sender = "user"
        response_chain = []  # Store all responses for ordered logging
        response_counter = 1  # Global response counter

        async for chunk in team.run_stream(task=task):
            if isinstance(chunk, TaskResult):
                continue
            
            response = chunk.content
            new_sender = chunk.source

            # Ensure response is a string
            if isinstance(response, list):
                response = "\n".join(map(str, response))  # Convert list to string
            
            # Generate unique numbered filename for each response
            file_name = f"{new_sender}_response_{response_counter}.txt"
            response_counter += 1  # Increment response counter

            # Log each response dynamically
            mlflow.log_metric(f"{new_sender} Response Length", len(response))
            mlflow.log_text(response, file_name)

            # Store response with sequence number for chat history
            response_chain.append(f"{response_counter-1}. [{new_sender}] {response}")
            add_message(new_sender, response)
        
        # Log entire conversation history in order
        chat_file_name = f"chat_history_{time.strftime('%Y%m%d-%H%M%S')}.txt"
        mlflow.log_text("\n".join(response_chain), chat_file_name)

        mlflow.log_param("End Time", time.strftime('%Y-%m-%d %H:%M:%S'))
        mlflow.log_param("Chat Duration seconds", time.time() - start_time)
        # End the MLflow run explicitly to prevent conflicts
        mlflow.end_run()

# Button click event
def on_click(event):
    asyncio.create_task(run_chat())

run_button.on_click(on_click)

# Layout
dashboard = pn.Column(
    pn.pane.Markdown("## 🤖 Single-Agent Chat", sizing_mode="stretch_width"),
    chat_feed,
    pn.Row(user_input, run_button)
)

# Serve the app
dashboard.servable()
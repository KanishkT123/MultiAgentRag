from autogen_agentchat.agents import AssistantAgent
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
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import TaskResult
import asyncio
import panel as pn

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.ERROR)

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

# Connect to CosmosDB
client = CosmosClient(COSMOS_URL, credential)
db = client.create_database_if_not_exists(DATABASE_NAME)

# Connect to OpenAI (for embedding)
openai_client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY, api_version="2023-05-15")

# Connect to Azure Chat Model (for all agents)
az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=AZURE_ORCH_MODEL,
    model=AZURE_ORCH_MODEL,
    api_version="2024-08-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_ad_token_provider=token_provider
)


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

def execute_sql(containerName:str, query:str):
    '''
    Execute a SQL query on the specified container in CosmosDB.
    Args:
        containerName (str): The name of the container to query.
        query (str): The SQL query to execute. Must be a read-only query.
    '''
    container = db.get_container_client(containerName)

    if not query.startswith("SELECT"):
        return "Error: Only read operations (SELECT queries) are allowed."

    results = []
    for item in container.query_items(
        query=query,
        enable_cross_partition_query=True  # Enable cross-partition querying
    ):
        results.append(item)
    
    return results if results else "No records found."

# Async function to retrieve input from UI
async def get_user_input(prompt: str, cancellation_token):
    while not user_input.value.strip():
        await asyncio.sleep(0.1)  # Wait for input asynchronously
    text = user_input.value.strip()
    user_input.value = ""  # Clear input field after reading
    return text

##########
# Agents #
##########

orchestrator = AssistantAgent(
    name= "orchestrator_agent",
    description = "An orchestrator agent that breaks down a complex query into subqueries. This agent should be the first to engage.",
    system_message = """
    You are an orchestration agent.
    Your job is to break down complex tasks into smaller, manageable subtasks amd assign tasks to other team members.
    Your team members are:
        - sql_agent: A helpful SQL query agent with access to CosmosDB. You assign structured data retrieval tasks to this agent. It has access to a database of fruits, species, rotting times, complaints, and current inventory. 
        - rag_agent: A helpful Retrieval Agent with access to VectorSearch. You assign unstructured data retrieval tasks to this agent. It has access to a database of fruits and complaints. The descriptions and complaints are both vectorized.

    These are all the containers in the attached CosmosDB:
        - Fruits: FruitID, Name, Color, Description, Season
        - FruitSpecies: FruitID,SpeciesID,SpeciesName
        - SpeciesRottingTime: SpeciesID,MinDaysToRot,MaxDaysToRot,SignsOfRot
        - Complaints: ComplaintID,SpeciesID,ComplaintDescription
        - Inventory: InventoryID,SpeciesID,QuantityOnHand,StorageLocation

    You do not directly execute queries yourself. Instead, post the broken-down tasks in the group chat, and let the relevant agent respond.
    Once all tasks are complete, summarize the findings and end with "TERMINATE".

    If you need to ask for further clarification, you can ask the user for more input.
    """,
    model_client=az_model_client
)

rag_agent = AssistantAgent(
    name = "rag_agent",
    description="A retrieval agent that retrieves unstructured data using Vector Search.",
    system_message = """You are a helpful Retrieval Agent. You retrieve fruit descriptions and complaints from a Cosmos DB with vector search enabled.
    You are an expert in retrieving relevant complaints, fruit descriptions, and unstructured data from Azure AI Search.
    Use tools to find the most relevant information based on the given query.
    You have access to the following containers:
    - Complaints: ComplaintID,SpeciesID,ComplaintDescription
    - Fruits: FruitID,Name,Color,Description,Season

    The descriptions and complaints are both vectorized.
    You can choose to return multiple fields if it will provide useful information.

    If the query is not related to unstructured text retrieval, let the SQL agent handle it.
    """,
    tools = [retrieve_results],
    reflect_on_tool_use=True,
    model_client=az_model_client
)

sql_agent = AssistantAgent(
    name = "sql_agent",
    description= "A SQL query agent that retrieves structured data from CosmosDB.",
    system_message = """
    You are a helpful SQL query agent. You convert natural language queries to SQL queries and execute them on CosmosDB.
    Your only job is to handle structured data retrieval tasks.
    You have access to the following 5 containers and columns:
    - Fruits: FruitID, Name, Color, Description, Season
    - FruitSpecies: FruitID,SpeciesID,SpeciesName
    - SpeciesRottingTime: SpeciesID,MinDaysToRot,MaxDaysToRot,SignsOfRot
    - Complaints: ComplaintID,SpeciesID,ComplaintDescription
    - Inventory: InventoryID,SpeciesID,QuantityOnHand,StorageLocation

    When responding:
    - **Convert natural language into SQL** using the schema.
    - These queries are executed through CosmosDB using the Azure Cosmos Python SDK.
    - **Execute the SQL query** and return results.
    - **Only query existing tables and columns**.
    - **Do not add or modify data**.
    - **ALL QUERIES MUST START WITH SELECT**
    - **NEVER RETRIEVE ALL RECORDS FROM ANY CONTAINER. THERE ARE TOO MANY ENTRIES FOR THIS.**

    An example of a valid query is "SELECT c.FruitID FROM c WHERE c.Color = 'Yellow'" for the Fruits container.

    Do not ask for clarification unless you must.
    """,
    model_client=az_model_client,
    tools=[execute_sql],
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
    [orchestrator, rag_agent, sql_agent, user_proxy],
    model_client=az_model_client,
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
    "orchestrator_agent": {"background-color": "#C8E6C9", "padding": "8px", "border-radius": "8px", "margin": "5px", "width": "fit-content", "align-self": "flex-end"},
    "sql_agent": {"background-color": "#FFECB3", "padding": "8px", "border-radius": "8px", "margin": "5px", "width": "fit-content"},
    "rag_agent": {"background-color": "#E1BEE7", "padding": "8px", "border-radius": "8px", "margin": "5px", "width": "fit-content"},  # Purple for RAG Agent
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
        return  # Do nothing if empty input

    add_message("user", task)  # Show user message
    user_input.value = ""  # Clear input field
    sender = 'user'

    async for chunk in team.run_stream(task=task):
        if type(chunk) is TaskResult:
            continue
        else:
            sender = chunk.source
            add_message(sender, chunk.content)

# Button click event
def on_click(event):
    asyncio.create_task(run_chat())

run_button.on_click(on_click)

# Layout
dashboard = pn.Column(
    pn.pane.Markdown("## 🤖 Multi-Agent Chat", sizing_mode="stretch_width"),
    chat_feed,
    pn.Row(user_input, run_button)
)

# Serve the app
dashboard.servable()
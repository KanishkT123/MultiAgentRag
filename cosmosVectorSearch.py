import os
from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import pandas as pd
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from tenacity import retry, stop_after_attempt, wait_random_exponential 
import logging

# Load environment variables from .env file
load_dotenv()

COSMOS_URL = os.getenv("COSMOS_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDING= os.getenv("AZURE_EMBEDDING")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
DIMENSIONS = 1536
# Authenticate using Managed Identity (DefaultAzureCredential)
credential = DefaultAzureCredential()

# Connect to CosmosDB
client = CosmosClient(COSMOS_URL, credential)
db = client.create_database_if_not_exists(DATABASE_NAME)

openai_client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY, api_version="2023-05-15")

# Create the vector embedding policy to specify vector details
vector_embedding_policy = {
    "vectorEmbeddings": [ 
        { 
            "path":"/embedding",
            "dataType":"float32",
            "distanceFunction":"cosine",
            "dimensions":DIMENSIONS
        }, 
    ]
}

indexing_policy = {
    "includedPaths": [ 
    { 
        "path": "/*" 
    } 
    ], 
    "excludedPaths": [ 
    { 
        "path": "/\"_etag\"/?",
        "path": "/" + "embedding" + "/*",
    } 
    ], 
    "vectorIndexes": [ 
        {
            "path": "/embedding", 
            "type": "quantizedFlat" 
        }
    ]
}

# Create a container with vector embedding policy
try:
    FruitContainer = db.create_container_if_not_exists(
        id="Fruits",
        partition_key=PartitionKey("/id"),
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_embedding_policy
    )

    print("Fruit Container created successfully")

    ComplaintContainer = db.create_container_if_not_exists(
        id="Complaints",
        partition_key=PartitionKey("/id"),
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_embedding_policy
    )

    print("Complaint Container created successfully")
except Exception as e:
    print("An error occurred:", e)
    raise

@retry(wait=wait_random_exponential(min=2, max=300), stop=stop_after_attempt(20))
def generate_embeddings(text):
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

# Function to upload CSV data to CosmosDB with optional embeddings
def upload_csv_to_cosmos(csv_file, collection_name, embedding_column=None):
    """
    Uploads a CSV file to CosmosDB and optionally generates embeddings.
    
    :param csv_file: Path to the CSV file
    :param collection_name: Name of the CosmosDB collection
    :param embedding_column: Column name to generate embeddings (Optional)
    """
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Get collection client
    collection = db.get_container_client(collection_name)

    # Iterate over rows and insert into CosmosDB
    for _, row in df.iterrows():
        data = row.to_dict()
        data["id"] = str(data.get("id", row.name))  # Assign a unique ID
        
        # Generate embeddings if a column is specified
        if embedding_column and embedding_column in data:
            data["embedding"] = generate_embeddings(data[embedding_column])

        # Insert into CosmosDB
        collection.create_item(body=data)

    print(f"Data from {csv_file} uploaded successfully to {collection_name}.")

upload_csv_to_cosmos("data/table1fruits.csv", "Fruits", embedding_column="Description")
upload_csv_to_cosmos("data/table4fruitspeciescomplaints.csv", "Complaints", embedding_column="ComplaintDescription")
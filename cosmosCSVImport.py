import os
import pandas as pd
from azure.cosmos import CosmosClient, exceptions
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

COSMOS_URL = os.getenv("COSMOS_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")

# Authenticate using Managed Identity (DefaultAzureCredential)
credential = DefaultAzureCredential()

# Connect to CosmosDB
client = CosmosClient(COSMOS_URL, credential)

# Ensure database exists
try:
    database = client.get_database_client(DATABASE_NAME)
except exceptions.CosmosResourceNotFoundError:
    print(f"Database '{DATABASE_NAME}' not found. Creating it now...")
    database = client.create_database(DATABASE_NAME)

def ensure_container_exists(collection_name, partition_key="/id"):
    """
    Ensures the specified collection exists in the Cosmos DB.
    If not, it creates the collection with a partition key.
    """
    try:
        return database.get_container_client(collection_name)
    except exceptions.CosmosResourceNotFoundError:
        print(f"❌ Collection '{collection_name}' not found. Creating it now...")
        return database.create_container(id=collection_name, partition_key={"paths": [partition_key]})

def upload_csv_to_cosmos(csv_file, collection_name, id_column):
    """
    Reads a CSV file and uploads the data to the specified Cosmos DB collection.
    """
    # Ensure collection exists
    collection = ensure_container_exists(collection_name)

    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Insert each row as a JSON document
    for _, row in df.iterrows():
        data = row.to_dict()
        data["id"] = str(data.get(id_column, row.name))  # Assign a unique ID
        try:
            collection.create_item(body=data)
        except exceptions.CosmosResourceExistsError:
            print(f"⚠️ Item with ID '{data['id']}' already exists in {collection_name}. Skipping...")

    print(f"✅ {csv_file} uploaded successfully to {collection_name}.")

# Upload FruitSpecies Data
upload_csv_to_cosmos("data/table2fruitspecies.csv", "FruitSpecies", "SpeciesID")

# Upload SpeciesRottingTime Data
upload_csv_to_cosmos("data/table3speciesrottingtime.csv", "SpeciesRottingTime", "SpeciesID")

# Upload Inventory Data
upload_csv_to_cosmos("data/table5inventory.csv", "Inventory", "InventoryID")

def query_collection(collection_name):
    """
    Queries a collection and prints the first few records.
    """
    collection = ensure_container_exists(collection_name)
    query = "SELECT * FROM c"
    items = list(collection.query_items(query, enable_cross_partition_query=True))
    
    print(f"🔍 {collection_name} Data Preview:")
    for item in items[:5]:  # Show first 5 records
        print(item)

# Verify each collection
query_collection("FruitSpecies")
query_collection("SpeciesRottingTime")
query_collection("Inventory")

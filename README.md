# Multi-Agent RAG Chat

This repository sets up a simple Multi-Agent RAG Chat with:

1. `Orchestrator` Agent - For task breakdown and decision making
2. `SQL` Agent - For structured SQL queries
3. `RAG` Agent - For unstructed RAG queries

The agents run in Autogen 0.4's `SelectorGroupChat`, which automatically selects the next speaker and helps the orchestrator handoff messages between agents.

This requires:

1. Azure CosmosDB with Vector Search enabled and Containers built
2. Azure AI Foundry instance with an Embedding deployment (Tested with `embedding-3-large`)
3. Azure AI Foundry instance with an inference model deployment (Tested with `gpt-4o`)

To Run:

1. Run `CosmosCSVImport.py`
2. Run `CosmosVectorSearch.py`
3. Use panel to run `multiAgentChat.py` via `panel serve .\multiAgentChat.py`

More detailed instructions to follow. In the meantime, please look at the following links for more info:
1. https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/selector-group-chat.html
2. https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/quickstart.html
3. https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search

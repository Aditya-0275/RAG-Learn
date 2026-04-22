import json
import operator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document 
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Dict, TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, FunctionMessage
from langgraph.prebuilt import ToolInvokation
from dotenv import load_dotenv

load_dotenv()
hub = Client()

# Retriever
urls = [
	"https://lilianweng.github.io/posts/2023-06-23-agent/",
	"https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
	"https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
	chunk_size = 250, 
	chunk_overlap = 0
)

doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
	documents = doc_splits,
	collection_name = "rag-chroma",
	embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

retriever = vectorstore.as_retriever()

"""
State -
We will define a graph.
Our state will be a dict.
We can access it from any graph nodes state['keys']
"""

class GraphState(TypedDict):
	"""
		Represents the state of an agent in the conversation.

		Attributes:
			keys: A dictionary where each key is a string ans the value is expected to be a list or another string that supports addition with `operator.add`. This could be used, for instance, to accumulate metadata or other pieces of data throughout the graph.
	"""

	keys: Dict[str, any]

"""
Nodes and Edges - 
Each node will simply modify the state
Each edge will choose which node to choose next

"""



















































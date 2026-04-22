import os
from dotenv import load_dotenv
from langchainhub import Client

hub = Client()
load_dotenv()
os.environ.setdefault("USER_AGENT", "adaptive-rag/1.0")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Literal
import langchain

# Data Model
class web_search(BaseModel):
	"""
		The internet. Use web search for questions that are related to anything else than agents, prompt engineering and adversarial attacks.
	"""
	query: str = Field(description="The query to use when searching the internet.")

class vectorstore_search(BaseModel):
	"""
		A vectorstore containing the documents related to agents, prompt engineering ans adversarial attacks. Use the vectorstore for questions on these topics.
	"""
	query: str = Field(description="The query to use when searching the vectorstore.")

# For Grader 
class GradeDocuments(BaseModel):
	"""
		Binary score for relevance check on retrieved documents.
	"""
	binary_score: Literal["yes", "no"] = Field(description="Relevance label: 'yes' or 'no'")

# Preamble
preamble = """
	You are an expert at routing a user question to a vectorstore or web search.
	The vectorstore contains documents related to agents, prompt engineering and adversarial attacks.
	Use the vectorstore for questions on these topics. Otherwise use web search.
"""

# Set embeddings
embd = CohereEmbeddings(model="embed-english-v3.0")

# Docs to index
urls = [
	"https://lilianweng.github.io/posts/2023-06-23-agent/",
	"https://lilianweng.github.io/posts/2023-03-15-prompt-engineering",
	"https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm"
]

# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split 
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
	chunk_size = 512,
	chunk_overlap = 0
)

doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorstore
vectorstore = Chroma.from_documents(
	documents = doc_splits,
	embedding = embd
)

retriever = vectorstore.as_retriever()

# print(retriever.invoke("Agent memory"))

# llm with tool use and preamble 
llm = ChatCohere(model = "command-r-08-2024", temperature = 0)
structured_llm_router = llm.bind_tools(tools = [web_search, vectorstore_search], preamble = preamble)

# Prompt
route_prompt = ChatPromptTemplate.from_messages(
	[
		("human", "{question}")
	]
)

# Test
question_router = route_prompt | structured_llm_router

# response = question_router.invoke({"question": "Who will the Bears draft first in the NFL draft?"})
# print(response.response_metadata['tool_calls'])

# response = question_router.invoke({"question": "What are the types of agent memory?"})
# print(response.response_metadata['tool_calls'])

# response = question_router.invoke({"question": "Hi, How are you?"})
# print('tool_calls' in response.response_metadata)


preamble_grader = """
	You are a grader assessing relevance of retrieved document to a user question. \n
	If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
	Return only one label: 'yes' or 'no' to indicate whether the document is relevant to the question.
"""

structured_llm_grader = llm.with_structured_output(GradeDocuments, preamble = preamble_grader)

grade_prompt = ChatPromptTemplate.from_messages(
	[
		("human", "Retrieved Document: \n\n {document} \n\n User question: {question}" )
	]
)

retrieval_grader = grade_prompt | structured_llm_grader
question = "types of agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
response = retrieval_grader.invoke({"question": question, "document": doc_txt})
# print(f"This is the response for grader test: \n\n {response}")

# Generation
preamble_generation = """
You are an assistant for question-answering tasks. Use the following following pieces of retrieved context to answer the question.
If you do not know the answer just say - I do not know.
Use three sentences maximum to keep the answer consize.
"""

prompt = lambda x: ChatPromptTemplate.from_messages(
	[
		HumanMessage(
			f"Question: {x['question']} \n Answer: ",
			additional_kwargs = {"documents": x["documents"]}
		)
	]
)

# Chain
rag_chain = (
	prompt | llm | StrOutputParser()
)

# Run
# generation = rag_chain.invoke({"documents": docs, "question": question})
# print(f"\n\n{generation}")


llm = llm.bind(preamble = preamble_generation)

prompt = lambda x: ChatPromptTemplate.from_messages(
	[
		HumanMessage(
			f"Question: {x['question']} \n Answer: "
		)
	]
)

llm_chain = ( prompt | llm | StrOutputParser() )

question = "Hi, How are you?"
generation = llm_chain.invoke({"question": question})
print(generation)












































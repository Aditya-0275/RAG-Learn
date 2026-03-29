import bs4
import tiktoken
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchainhub import Client
hub = Client()
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# Test code here
question = "What sport do I like to play?"
document = "I love play foorball with my friends on weekends."

def _num_tokens_from_string_(string: str, encoding_name: str) -> int:
	""" Returns number of tokens in a text string. """
	encoding = tiktoken.get_encoding(encoding_name)
	num_tokens = len(encoding.encode(string))
	return num_tokens

def _cosine_similarity_(vec1, vec2):
	""" Returns cosine similarity between two vectors. """
	dot_product = np.dot(vec1, vec2)
	norm_vec1 = np.linalg.norm(vec1)
	norm_vec2 = np.linalg.norm(vec2)
	return dot_product / (norm_vec1*norm_vec2)

print(f"TEST: The number of tokens in the question string: {_num_tokens_from_string_(question, 'cl100k_base')}")

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
query_embd_result = embedding.embed_query(question)
doc_embd_result = embedding.embed_query(document)
print(f"TEST: This is the length of ques embd: {len(query_embd_result)}")
print(f"TEST: This is the length of doc embd: {len(doc_embd_result)}")

similarity = _cosine_similarity_(query_embd_result, doc_embd_result)
print(f"TEST: Cosine similarity: {similarity}")

""" Indexing starts here. """

# Load the blog
loader = WebBaseLoader(
		web_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",),
		bs_kwargs = dict(
				parse_only = bs4.SoupStrainer(
						class_ = ("post-content", "post-title", "post-header")
					)
			),
	)
blog_docs = loader.load()

# Split the docs 
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
		chunk_size = 300, 
		chunk_overlap = 50
	)
splits = text_splitter.split_documents(blog_docs)

# Indexing the splitted docs
vectorstore = Chroma.from_documents(
		documents = splits,
		embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
	)
# k = 1 means that we want k nearest neighbours related to out query in the embedding space.
retriever = vectorstore.as_retriever(search_kwargs = {"k": 1})

# Test the retriever
# test_doc = retriever.invoke("What is Task Decomposition?")
# print(f"Retriever fetched the test doc of length: {len(test_doc)}, Content: ")
# print(test_doc)

""" Generation starts here. """

# Prompt 
template = """
Answer the question based on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
print(f"TEST: Here is the prompt: {prompt}")

# Define llm
llm = ChatGroq(
		model = "llama-3.1-8b-instant",
		temperature = 0
	)

# Chain example 
chain = prompt | llm
print(f"Here is invoke event: ")
AIMessage = chain.invoke({"context": splits[:10],"question": "What is Task Decomposition?"})
print(f"{AIMessage}")

# heading to make a rag
prompt_hub_rag = hub.pull("rlm/rag-prompt")
print(f"The prompt_hub_rag var: {prompt_hub_rag}")

rag_chain = (
		{"context": retriever, "question": RunnablePassthrough()}
		| prompt
		| llm
		| StrOutputParser()
	)

AIMsgUsingAutomatedRAG = rag_chain.invoke("What is Task Decomposition?")
print(f"Message using RAG: {AIMsgUsingAutomatedRAG}")








































import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import dumps, loads
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()

def _get_unique_union_(documents: list[list]):
	""" Unique union of retrieved docs. """
	flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
	unique_docs = list(set(flattened_docs))
	return [loads(doc) for doc in unique_docs]

loader = WebBaseLoader(
		web_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",),
		bs_kwargs = dict(
				parse_only = bs4.SoupStrainer(
						class_ = ("post-content", "post-title", "post-header")
					)
			),
	)
blog_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
		chunk_size = 300,
		chunk_overlap = 50
	)

splits = text_splitter.split_documents(blog_docs)

vectorstore = Chroma.from_documents(
		documents = splits,
		embedding = HuggingFaceEmbeddings()
	)


retriever = vectorstore.as_retriever()

template = """
You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance based similarity search. Provide these alternative questions separated by new lines.
Original Question: {question}
"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
		prompt_perspectives
		| ChatGroq(model = "llama-3.1-8b-instant", temperature = 0)
		| StrOutputParser()
		| (lambda x : x.split("\n"))
	)

question = "What is task decomposition for llm agents?"
retrieval_chain = generate_queries | retriever.map() | _get_unique_union_
docs = retrieval_chain.invoke({"question": question})
print(f"The length of docs is: {len(docs)}")
print(f"Retrieved docs: {docs}")

# RAG
template = """
Answer the following question based on the context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatGroq(model = "llama-3.1-8b-instant", temperature = 0)

final_rag_chain = (
		{"context": retrieval_chain, "question": itemgetter("question")}
		| prompt
		| llm
		| StrOutputParser()
	)

print(f"The final answer to the question is: {final_rag_chain.invoke({"question": question})}")


print(f"\n   This is the Rag-Fusion part.   \n")

template = """
You are a helpful assistant that generates multiple search queries based on single input query. \n
Generate multiple search queries related to: {question}
Output (4 queries):
"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)
generate_queries = (
		prompt_rag_fusion
		| ChatGroq(model = "llama-3.1-8b-instant", temperature = 0)
		| StrOutputParser()
		| (lambda x: x.split("\n"))
	)

def _reciprocal_rank_fusion_(results: list[list], k = 60):
	"""
	Reciprocal_Rank_Fusion that takes multiple lists of ranked documents and an optional parameter k used in the RRF formula.
	"""
	fused_scores = {}

	for docs in results:
		for rank, doc in enumerate(docs):
			doc_str = dumps(doc)
			if doc_str not in fused_scores:
				fused_scores[doc_str] = 0
			prev_score = fused_scores[doc_str]
			# Update the score of the document using the RRF formula : 1 / (rank + k)
			fused_scores[doc_str] += 1 / (rank + k)


	# Sort the documents based on their fused scores in descending order to get the final ranked results
	reranked_results = [
		(loads(doc), score)
		for doc, score in sorted(fused_scores.items(), key = lambda x: x[1], reverse = True)
	]

	return reranked_results

retrieval_chain_rag_fusion = generate_queries | retriever.map() | _reciprocal_rank_fusion_
docs = retrieval_chain_rag_fusion.invoke({"question": question})
print(f"The length of the rag fustion retrieved docs is: {len(docs)}. \n The docs looks like -> \n {docs} \n END OF DOCS")

# RAG
template = """
Answer the following question based on this context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
		{"context": retrieval_chain_rag_fusion, "question": itemgetter("question")}
		| prompt
		| llm
		| StrOutputParser()
	)

print(f"The final answer for the question using Rag-Fusion method: \n {final_rag_chain.invoke({"question": question})}")

print("---- END OF SCRIPT ----")












































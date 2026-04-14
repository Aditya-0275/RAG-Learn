import bs4
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
	Please write a scientific paper passage to answer the question\n
	Question: {question}\n
	Passage:\n
"""

llm = ChatGroq(model = "llama-3.1-8b-instant", temperature = 0)

prompt_hyde = ChatPromptTemplate.from_template(template)

generate_docs_for_retrieval = prompt_hyde | llm | StrOutputParser()

question = "What is Task Decomposition for llm agents?"

retriever_chain = generate_docs_for_retrieval | retriever
retrieved_docs = retriever_chain.invoke({"question":question}) 

template = """

Answer the following question based on the following context:\n

Context: {context}

Question: {question}\n

"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
		prompt
		| llm
		| StrOutputParser()
)

result = final_rag_chain.invoke({"context":retrieved_docs, "question":question}) 

print(result)

















































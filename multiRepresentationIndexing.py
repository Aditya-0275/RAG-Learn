import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage import InMemoryByteStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-02-05-human-data-quality/")
docs.extend(loader.load())
MAX_DOC_CHARS = 3500
MAX_CONCURRENCY = 1
BATCH_SIZE = 1

docs = [
	Document(page_content=doc.page_content[:MAX_DOC_CHARS], metadata=doc.metadata)
	for doc in docs
]

chain = (
	{"doc": lambda x: x.page_content}
	| ChatPromptTemplate.from_template("Summarise the following document:\n\n{doc}")
	| ChatGroq(model = "llama-3.1-8b-instant", max_retries = 0)
	| StrOutputParser()
)

summaries = []
for i in range(0, len(docs), BATCH_SIZE):
	current_batch = docs[i : i + BATCH_SIZE]
	batch_summaries = chain.batch(current_batch, {"max_concurrency": MAX_CONCURRENCY})
	summaries.extend(batch_summaries)
summaries = chain.batch(docs, {"max_concurrency": 5})

for idx, summary in enumerate(summaries, start=1):
	print(f"Summary {idx}:\n{summary}\n")

# The vectorstore used to index the child chunks.
vectorstore = Chroma(collection_name = "summaries", embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

#  Storage layer for the parent documents.
store = InMemoryByteStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
	vectorstore = vectorstore,
	byte_store = store,
	id_key = id_key
)

doc_ids = [str(uuid.uuid4()) for _ in docs]

# Docs linked to summaries
summary_docs = [
	Document(page_content=s, metadata = {id_key: doc_ids[i]})
	for i, s in enumerate(summaries)
]

# Add to retriever 
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

query = "Memory in agents"
sub_docs = vectorstore.similarity_search(query, k = 1)
print(sub_docs[0])

retrieved_docs = retriever.invoke(query)
print(retrieved_docs[0].page_content[0:500])















































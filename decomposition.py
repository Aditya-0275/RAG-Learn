import bs4
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from operator import itemgetter


load_dotenv()

template = """
You are a helpful assistant that generates multiple sub-questions related to input question. \n
The goal is to breakdown the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to the question (Just give me the queries not the explaination or headings in pointers): {question} \n
Output three queries:
"""

prompt_decomposition = ChatPromptTemplate.from_template(template)

llm = ChatGroq(model = "llama-3.1-8b-instant", temperature = 0)

generate_queries_decoposition = (prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

question = "What are the main components of llm-powered autonomous agents system?"

sub_questions = generate_queries_decoposition.invoke({"question": question})

print(f"The sub-questions are: \n {sub_questions} \n End of sub_questions")

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
Here is the question you need to answer:
\n --- \n {question} \n --- \n
Here is the additional context related to the question:
\n --- \n {context} \n --- \n
Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)

def _format_qa_pair_(question, answer):
	""" Format Q and A pair """
	formatted_string = ""
	formatted_string += f"Question: {question} \nAnswer: {answer}\n\n"
	return formatted_string.strip()

qa_pairs = ""

for q in sub_questions:
	rag_chain = (
			{"context": itemgetter("question") | retriever, "question": itemgetter("question") , "qa_pairs": itemgetter("qa_pairs")}
			| decomposition_prompt
			| llm 
			| StrOutputParser()
		)
	answer = rag_chain.invoke({"question":q, "qa_pairs":qa_pairs})
	qa_pair = _format_qa_pair_(q, answer)
	qa_pairs = qa_pairs + "\n --- \n" + qa_pair
print(f"\n ---------- \n")
print(f"{answer}")


# Can also get the answers of the sub_queries parallely and then pass those answers as context to the llm along with retrieved docs for some usecases.
















































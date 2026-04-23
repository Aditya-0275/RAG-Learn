import os
from dotenv import load_dotenv
from langchainhub import Client

hub = Client()
load_dotenv()
os.environ.setdefault("USER_AGENT", "adaptive-rag/1.0")

from langgraph.graph import END, StateGraph, START
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import Literal, TypedDict, List
from pprint import pprint
import langchain
import time
import httpx
import httpcore

TIMEOUT_EXCEPTIONS = (httpx.ReadTimeout, httpcore.ReadTimeout)

def invoke_with_retry(chain, payload, max_attempts=3):
	for attempt in range(1, max_attempts + 1):
		try:
			return chain.invoke(payload)
		except TIMEOUT_EXCEPTIONS:
			if attempt == max_attempts:
				raise
			wait_seconds = 2 ** (attempt - 1)
			print(
				f"---REQUEST TIMEOUT: retrying in {wait_seconds}s (attempt {attempt}/{max_attempts})---"
			)
			time.sleep(wait_seconds)

class GraphState(TypedDict):
	"""
		Represents the state of our graph:

		Attributes
		 - question: question
		 - generation: LLM generation
		 - documrnts: List of documents
	"""
	question: str
	generation: str
	documents: List[str]

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
	""" Binary score for relevance check on retrieved documents. """
	binary_score: Literal["yes", "no"] = Field(description="Relevance label: 'yes' or 'no'")

# Hallucianation grader 
class GradeHallucinations(BaseModel):
	""" Binary score for halluciantion present in generation answer. """
	binary_score: Literal["yes", "no"] = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
	""" Binary score to assess answer addresses the question. """
	binary_score: Literal["yes", "no"] = Field(description="Answer addresses the question, 'yes' or 'no'")

# Preamble
preamble = """
	You are an expert at routing a user question to a vectorstore or web search.
	The vectorstore contains documents related to agents, prompt engineering and adversarial attacks.
	Use the vectorstore for questions on these topics. Otherwise use web search.
"""

# Set embeddings
embd = CohereEmbeddings(
	model = "embed-english-v3.0",
	request_timeout = 120,
	max_retries = 3
)

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
llm = ChatCohere(model = "command-r-08-2024", temperature = 0, timeout_seconds = 120)
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
response = invoke_with_retry(retrieval_grader, {"question": question, "document": doc_txt})
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
generation = invoke_with_retry(rag_chain, {"documents": docs, "question": question})
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
generation = invoke_with_retry(llm_chain, {"question": question})
print(generation)

preamble_hallucinations = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Give a binary score 'yes' or 'no', 'yes' means the answer is grounded in / supported by the set of facts.
"""

llm = ChatCohere(model = "command-r-08-2024", temperature = 0, timeout_seconds = 120)
structured_llm_grader = llm.with_structured_output(GradeHallucinations, preamble = preamble_hallucinations)

hallucinations_prompt = ChatPromptTemplate.from_messages(
	[
		("human", "Set of facts: \n\n{documents} \n\nLLM generation: \n\n{generation}")
	]
)

hallucination_grader = hallucinations_prompt | structured_llm_grader
print(invoke_with_retry(hallucination_grader, {"documents": docs, "generation": generation}))

preamble_answer_grader = """
You are a grader assessing whether an answer addresses / resolves a question. \n
Give a binary score 'yes' or 'no', 'yes' means the answer resolves the question.
"""

structured_llm_grader = llm.with_structured_output(GradeAnswer, preamble = preamble_answer_grader)

answer_prompt = ChatPromptTemplate.from_messages(
	[
		("human", "User question: \n\n{question} \n\nLLM generation: {generation}")
	]
)

answer_grader = answer_prompt | structured_llm_grader
print(invoke_with_retry(answer_grader, {"question": question, "generation": generation}))
question_rewriter_preamble = """
You are a question re-writer that converts a user question into a better version for vectorstore retrieval.
"""

question_rewriter_prompt = ChatPromptTemplate.from_messages(
	[
		("human", "Initial question:\n\n{question}\n\nRewrite this question so it is optimized for semantic retrieval.")
	]
)

question_rewriter_llm = ChatCohere(
	model = "command-r-08-2024",
	temperature = 0,
	timeout_seconds = 120
).bind(preamble = question_rewriter_preamble)

question_rewriter = question_rewriter_prompt | question_rewriter_llm | StrOutputParser()


web_search_tool = None

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = invoke_with_retry(rag_chain, {"documents": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = invoke_with_retry(
            retrieval_grader,
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    web_search = "Yes" if len(filtered_docs) == 0 else "No"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = invoke_with_retry(question_rewriter, {"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    global web_search_tool
    if web_search_tool is None:
        try:
            web_search_tool = TavilySearchResults()
        except Exception as exc:
            raise RuntimeError(
                "Web search was requested but TAVILY_API_KEY is not configured."
            ) from exc
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search_node", web_search)  # web search

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# Run
# {"question": "How does the AlphaCodium paper work?"}
inputs = {"question": "What are the types of agent memory?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])


































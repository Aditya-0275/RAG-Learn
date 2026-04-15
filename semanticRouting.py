from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

physics_template = """
	You are a very smart physics professor. \n
	You are great at answering questions about physics in a consise and easy to understand manner. \n
	When you do not know the answer to the question you admit that you do not know. \n
	Here is a question: \n
	- {query} -
"""

math_template = """
	You are a very good mathematician. You are great at answering questions. \n
	You are so good because you are able to break down hard problems into their component parts, \n
	answer the component parts, and then put them together to answer the broader question.\n
	Here is the question: \n
	- {query} - 
"""

# Embed prompts
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

#  Route question to prompt

def prompt_router(input):
	query_embedding = embeddings.embed_query(input["query"])
	similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
	most_similar = prompt_templates[similarity.argmax()]

	print("Using Math" if most_similar == math_template else "Using Physics")

	return PromptTemplate.from_template(most_similar)


chain = (
	{"query": RunnablePassthrough()}
	| RunnableLambda(prompt_router)
	| ChatGroq(model = "llama-3.1-8b-instant", temperature = 0)
	| StrOutputParser()
)

print(chain.invoke("What's a Black Hole?"))
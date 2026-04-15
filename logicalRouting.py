from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Data Model
class RouteQuery(BaseModel):
	"""
		Route the user query to the most relevant datasource.
	"""

	datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
		...,
		description = "Given a user question choose which datasource would be most relevant for answering their question",
	)

# LLM with functional call
llm = ChatGroq(
		model = "llama-3.1-8b-instant",
		temperature = 0
	)

structured_llm = llm.with_structured_output(RouteQuery)


# Prompt

system = """

	You are an expert at routing a user question to the appropriate data source.
	Based on the programming language the question is referring to, route it to the relevant data source.

"""

prompt = ChatPromptTemplate.from_messages(
		[
			("system",system),
			("human","{question}"),
		]
	)


# router 
router = prompt | structured_llm

question = """

	Why doesn't the following code work:

		from langchain_core.prompts import ChatPromptTemplate

		prompt = ChatPromptTemplate.from_messages(
			["human", "speak in {language}"]
		)

		prompt.invoke("french")

"""

result = router.invoke({"question":question})

print(f"result: {result}, type: {type(result)}")


def choose_route(result):
	if "python_docs" in result.datasource.lower():
		return "chain_for_python_docs"
	elif "js_docs" in result.datasource.lower():
		return "chain_for_js_docs"
	else:
		return "chain_for_golang_docs"


full_chain = router | RunnableLambda(choose_route)

final_result = full_chain.invoke({"question":question})

print(f"Final result of logical route chain: {final_result}")


















































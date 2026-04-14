import bs4
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

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

examples = [
	{
		"input": "Could the members of the Police perform lawful arrests?",
		"output": "What can the members of The Police do?"
	},
	{
		"input": "Jan Sindel's was born in what country?",
		"output": "What is Jan Sindel's personal history?"
	}
]

# Now tranform these to example messages

example_prompt = ChatPromptTemplate.from_messages(
		[
			("human", "{input}"),
			("ai", "{output}")
		]
	)

few_shot_prompt = FewShotChatMessagePromptTemplate(
		example_prompt=example_prompt,
		examples=examples
	)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
				You are an expert at world knowledge.

				Your task is to generate a **step-back question**.

				A step-back question is a **more generic question that helps answer the original question.**

				Rules:
				- Rewrite the question into a **more general question**
				- The new question should help reason about the original question
				- Return **ONLY ONE question**
				- Do NOT add explanations
				- Do NOT add extra text
			"""
        ),
        few_shot_prompt,
        ("human", "{question}")
    ]
)

generate_queries_step_back = prompt | ChatGroq(model = "llama-3.1-8b-instant", temperature = 0) | StrOutputParser()
question = "What is Task Decomposition for llm agents?"
print(f"Generated: {generate_queries_step_back.invoke({'question' : question})}")

response_prompt_template = """
	You are an expert in world knowledge. I am going to ask you a question. Your response should be comprehensive and more relevant to the normal context, you can use step back context for more information.

	# normal context : {normal_context}
	# step back context : {step_back_context}

	# Orignal Question : {question}
	# Answer:
"""

response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain = (
	{
		"normal_context" : RunnableLambda(lambda x: x['question']) | retriever,
		"step_back_context" : generate_queries_step_back | retriever,
		"question" : lambda x: x['question']
	}
	| response_prompt
	| ChatGroq(model = "llama-3.1-8b-instant", temperature = 0)
	| StrOutputParser()
)

print(chain.invoke({"question": question}))






















































from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import Optional, Literal, Tuple
from pydantic import BaseModel, Field, model_validator
from yt_dlp import YoutubeDL
import datetime
from dotenv import load_dotenv

load_dotenv()

VIDEO_URL = "https://www.youtube.com/watch?v=sVcwVQRHIc8&list=WL&index=4"

def fetch_video_info(video_url: str) -> dict:
	ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
	with YoutubeDL(ydl_opts) as ydl:
		info = ydl.extract_info(video_url, download=False)
	return {
		"title": info.get("title", "Unknown"),
		"description": info.get("description", "Unknown"),
		"view_count": info.get("view_count", 0),
		"thumbnail_url": info.get("thumbnail"),
		"publish_date": info.get("upload_date"),
		"length": info.get("duration", 0),
		"author": info.get("uploader", "Unknown"),
		"channel_id": info.get("channel_id"),
		"webpage_url": info.get("webpage_url", video_url),
	}

docs = YoutubeLoader.from_youtube_url(
	VIDEO_URL,
	add_video_info=False
).load()

video_info = fetch_video_info(VIDEO_URL)
for doc in docs:
	doc.metadata.update(video_info)

print(docs[0].metadata)


"""
Let's assume we have built an index that:

 1. Allows us to perform unstructured search over the contents and title of each document 
 2. And use range filtering on view count, publication date and length.

We want to convert natural language into structured search queries.
We can define a schema for structured search queries.
"""

class TutorialSearch(BaseModel):
	"""
		Search over a database of tutorial videos about a software library.
	"""

	content_search: str = Field(
		... , 
		description = "Similarity search query applied to video transcripts.",
	)

	title_search: Optional[str] = Field(
		None,
		description = (
			"Alternative version of the content search query to apply to video titles. "
			"Should be succinct and only include key words that could be in a video "
			"title."
		)
	)

	min_view_count: Optional[int] = Field(
		None,
		description = "Minimum view count filter, inclusive. Only use if explicitly specified."
	)

	max_view_count: Optional[int] = Field(
		None,
		description = "Maximum view count filter, exclusive. Only use if explicitly specified."
	)

	earliest_publish_date: Optional[datetime.date] = Field(
		None,
		description = "Earliest publish date filter, inclusive. Only use if explicitly specified."
	)

	latest_publish_date: Optional[datetime.date] = Field(
		None,
		description = "Latest publish date filter, exclusive. Only use if explicitly specified."
	)

	min_length_section: Optional[int] = Field(
		None,
		description = "Minimum video length in seconds, inclusive. Only use if explicitly specified."
	)

	max_length_section: Optional[int] = Field(
		None,
		description = "Maximum video length in seconds, exclusive. Only use if explicitly specified."
	)
	@model_validator(mode="after")
	def fill_title_search_if_missing(self) -> "TutorialSearch":
		if self.title_search is None:
			self.title_search = self.content_search
		return self

	def pretty_print(self) -> None:
		for field in type(self).model_fields:
			if getattr(self, field) is not None and getattr(self, field) != getattr(type(self).model_fields[field], "default", None):
				print(f"{field}: {getattr(self, field)}")


# Now we prompt llm to produce queries

system = """
	You are an expert at converting user questions into database queries. \n
	You have access to a database of tutorial videos about a software library for building LLM powered applications. \n
	Given a question return a database query optimized to retrieve the most relevant results. \n

	If there are acronyms or words you are not familier with, do nnot try to rephrase them.
"""

prompt = ChatPromptTemplate.from_messages(
	[
		("system", system),
		("human", "{question}")
	]
)

llm = ChatGroq(
		model = "llama-3.1-8b-instant",
		temperature = 0
	)

structured_llm = llm.with_structured_output(TutorialSearch)
query_analyser = prompt | structured_llm

parsed_query = query_analyser.invoke({"question": "rag from scratch"})
parsed_query.pretty_print()

print(f"First query: {parsed_query}")

second_query = query_analyser.invoke({"question": "videos on chat langchain published in 2023"})
second_query.pretty_print()
print(f"Second query: {second_query}")





























































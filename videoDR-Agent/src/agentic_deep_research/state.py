"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from tkinter import image_names
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The specific query or question to research via internet search. Should be a precise, targeted question, described in high detail (at least a paragraph), to guide effective web searches.",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""
    key_answer: str = Field(
        description="A concise keyword answer to the user question, based on the research findings.",
    )

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

class VideoInfoOutput(BaseModel):
    """Model for video information output."""
    
    video_info: str = Field(
        description="A detailed and specific description of the video content related to user questions.",
    )
    reasoning: str = Field(
        description="The reasoning why you want to describe the video content in this way.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class Source(BaseModel):
    """Reference source with optional note."""
    title: str = Field(description="Title or label of the source")
    url: str = Field(description="URL of the source")
    note: str | None = Field(default=None, description="Optional note about how the source was used")

class FinalAnswerOutput(BaseModel):
    """Structured final answer output."""
    final_answer: str = Field(description="Concise keyword answer for the user")

class FinalReportOutput(BaseModel):
    """Structured final report output."""
    final_report: str = Field(description="Full markdown report with sections and Sources")
    references: list[Source] = Field(description="List of sources referenced in the report")
    final_answer: str = Field(description="Concise keyword answer for the user")


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""
    video_url: Optional[str] = None
    images_url: list[str] = []

class SupervisorState(MessagesState):
    """State for the supervisor that manages research tasks."""
    
    # supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    video_url: Optional[str]
    images_url: list[str] = []
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

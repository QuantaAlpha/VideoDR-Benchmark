import os
from typing import Literal

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agentic_deep_research.configuration import Configuration
from agentic_deep_research.nodes.supervisor import supervisor
from agentic_deep_research.nodes.supervisor_tools import supervisor_tools
from agentic_deep_research.prompts import (
    get_video_info_instructions,
    lead_researcher_prompt,
)
from agentic_deep_research.state import (
    AgentInputState,
    SupervisorState,
    VideoInfoOutput,
)
from agentic_deep_research.utils import (
    configurable_model,
    get_api_key_for_model,
    get_today_str,
)


async def get_video_infomation(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor"]]:
    configurable = Configuration.from_runnable_config(config)
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": configurable.research_model_base_url,
        "tags": ["langsmith:nostream"]
    }

    query_generator_model = (
        configurable_model
        .with_structured_output(VideoInfoOutput, method="json_schema")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    video_url = state.get("video_url", None)
    images_url = state.get("images_url", [])
    assert video_url is not None or images_url, "Either video_url or images_url must be provided"

    messages = state.get("messages", [])
    user_query = messages[-1].content
    
    if video_url:
        model_input = [
            SystemMessage(content=get_video_info_instructions.format(date=get_today_str())),
            HumanMessage(content=[
                {"type": "text", "text": user_query},
                {"type": "video_url", "video_url": {"url": video_url}}
            ])
        ]
    elif images_url:
        model_input = [
            SystemMessage(content=get_video_info_instructions.format(date=get_today_str())),
            HumanMessage(content=[
                {"type": "text", "text": user_query},
                *[{"type": "image_url", "image_url": {"url": image_url}} for image_url in images_url]
            ])
        ]
    response = await query_generator_model.ainvoke(model_input)
    user_query_template = '''<User Question>
{user_query}
</User Question>

<Video Infomation>
{video_info}
</Video Infomation>

<Video Infomation Reasoning>
{video_info_reasoning}
</Video Infomation Reasoning>
'''
            
    instruction_content = lead_researcher_prompt.format(
        date=get_today_str(),
        max_researcher_iterations=configurable.max_researcher_iterations,
        max_concurrent_research_units=configurable.max_concurrent_research_units

    )
    # Construct the messages with system instructions and video content
    new_messages = [
        SystemMessage(content=instruction_content),
        HumanMessage(content=user_query_template.format(
            user_query=user_query,
            video_info=response.video_info,
            video_info_reasoning=response.reasoning
        ))
    ]

    delete_operations = [RemoveMessage(id=m.id) for m in state["messages"]]
    
    return Command(
        goto="supervisor",
        update={
            "messages": delete_operations + new_messages
        }
    )

async def format_supervisor_input(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor"]]:
    """Format input messages for the supervisor with video context."""
    configurable = Configuration.from_runnable_config(config)
    video_url = state.get("video_url", None)
    images_url = state.get("images_url", [])
    assert video_url is not None or images_url, "Either video_url or images_url must be provided"

    messages = state.get("messages", [])
    user_query = messages[-1].content
            
    instruction_content = lead_researcher_prompt.format(
        date=get_today_str(),
        max_researcher_iterations=configurable.max_researcher_iterations,
        max_concurrent_research_units=configurable.max_concurrent_research_units

    )
    if video_url:
        new_messages = [
            SystemMessage(content=instruction_content),
            HumanMessage(content=[
                {"type": "text", "text": user_query},
                {"type": "video_url", "video_url": {"url": video_url}}
            ])
        ]
    elif images_url:
        new_messages = [
            SystemMessage(content=instruction_content),
            HumanMessage(content=[
                {"type": "text", "text": user_query},
                *[{"type": "image_url", "image_url": {"url": image_url}} for image_url in images_url]
            ])
        ]

    delete_operations = [RemoveMessage(id=m.id) for m in state["messages"]]
    
    return Command(
        goto="supervisor",
        update={
            "messages": delete_operations + new_messages
        }
    )


class DeepResearcher:
    def __init__(self, mode: Literal["agentic", "workflow"] = "agentic"):
        self.mode = mode
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(
            SupervisorState, 
            input=AgentInputState, 
            config_schema=Configuration
        )
        
        builder.add_node("supervisor", supervisor)
        builder.add_node("supervisor_tools", supervisor_tools)

        if self.mode == "agentic":
            builder.add_node("format_supervisor_input", format_supervisor_input)
            builder.add_edge(START, "format_supervisor_input")
            builder.add_edge("format_supervisor_input", "supervisor")
        elif self.mode == "workflow":
            builder.add_node("get_video_infomation", get_video_infomation)
            builder.add_edge(START, "get_video_infomation")
            builder.add_edge("get_video_infomation", "supervisor")
        else:
             raise ValueError(f"Invalid mode: {self.mode}. Must be 'agentic' or 'workflow'.")

        return builder.compile(checkpointer=MemorySaver())

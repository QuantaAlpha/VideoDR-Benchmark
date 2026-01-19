from logging import getLogger
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

from agentic_deep_research.configuration import Configuration
from agentic_deep_research.state import (
    ConductResearch,
    ResearchComplete,
    SupervisorState,
)
from agentic_deep_research.utils import (
    configurable_model,
    get_api_key_for_model,
    think_tool,
)

logger = getLogger(__name__)

async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    video_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": configurable.research_model_base_url,
        "tags": ["langsmith:nostream"]
    }
    research_iterations = state.get("research_iterations", 0)
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    if exceeded_allowed_iterations:
        lead_researcher_tools = [ResearchComplete, think_tool]
    else:
        # Available tools: research delegation, completion signaling, and strategic thinking
        lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    video_model = (
        configurable_model
        .bind_tools(lead_researcher_tools, tool_choice="auto")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(video_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    messages = state.get("messages", [])
    logger.warning(f"Supervisor input messages: {messages}")
    response = await video_model.ainvoke(messages)
    
    # Step 3: Update state and proceed to tool execution
    if not response.tool_calls:
        return Command(
            goto="supervisor",
            update={
                "messages": [response, HumanMessage(content="You must use the provided tools (ConductResearch, ResearchComplete, think_tool) to proceed. Do not ask for user input as this is an autonomous process.")],
                "research_iterations": research_iterations + 1
            }
        )

    return Command(
        goto="supervisor_tools",
        update={
            "messages": [response],
        }
    )

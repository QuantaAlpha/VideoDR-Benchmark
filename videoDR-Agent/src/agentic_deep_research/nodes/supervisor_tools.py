import asyncio
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command

from agentic_deep_research.configuration import Configuration
from agentic_deep_research.nodes.researcher import researcher_subgraph
from agentic_deep_research.state import SupervisorState
from agentic_deep_research.utils import get_notes_from_tool_calls, is_token_limit_exceeded


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])
    most_recent_message = messages[-1]
    
    # Define exit criteria for research phase
    no_tool_calls = not most_recent_message.tool_calls
    if no_tool_calls:
         # If no tool calls, it might be an intermediate message or error state.
         # Instead of raising ValueError, we return a command to go back to supervisor
         # This aligns with the fix in supervisor.py, acting as a failsafe.
         return Command(goto="supervisor")

    # Check if we've exceeded the maximum number of iterations significantly
    # This acts as a hard stop to prevent infinite loops of thinking
    research_iterations = state.get("research_iterations", 0)

    # Check if ResearchComplete is present - if so, we should allow it to proceed
    has_research_complete = any(tc["name"] == "ResearchComplete" for tc in most_recent_message.tool_calls)

    if research_iterations > configurable.max_researcher_iterations and not has_research_complete:
        tool_messages = []
        for tc in most_recent_message.tool_calls:
            tool_messages.append(ToolMessage(
                content="Maximum research iterations reached. Please provide your final answer now using the ResearchComplete tool based on the information you have gathered. Do not conduct any further research.",
                tool_call_id=tc["id"],
                name=tc["name"]
            ))
            
        return Command(
            goto="supervisor",
            update={
                "messages": tool_messages
            }
        )

    research_complete_tool_call = [
        tool_call["args"]["key_answer"]
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ResearchComplete"
    ]   
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if len(conduct_research_calls) > 0:
        # Only increment iterations if actual research was conducted
        research_iterations = state.get("research_iterations", 0) + 1
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # Handle research execution errors
            if is_token_limit_exceeded(e, configurable.research_model):
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(messages)
                    }
                )
            raise
    else:
        if len(research_complete_tool_call) != 0:
            return Command(
                goto=END,
                update={
                    "notes": get_notes_from_tool_calls(messages),
                    "messages": [AIMessage(content=research_complete_tool_call[0])]
                }
            )

    update_payload["messages"] = all_tool_messages
    update_payload["research_iterations"] = research_iterations
    return Command(
        goto="supervisor",
        update=update_payload
    )

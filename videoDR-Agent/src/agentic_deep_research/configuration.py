"""Configuration management for the Open Deep Research system."""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    TAVILY = "tavily"
    NONE = "none"

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    research_model_base_url: Optional[str] = Field(
        default="https://openrouter.ai/api/v1"
    )
    summarization_model_base_url: Optional[str] = Field(
        default="https://openrouter.ai/api/v1"
    )
    research_model: str = Field(
        default="openai:qwen/qwen3-vl-30b-a3b-instruct"
    )
    research_model_max_tokens: int = Field(
        default=81920,
    )
    summarization_model: str = Field(
        default="openai:qwen/qwen3-30b-a3b-instruct-2507",
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
    )
    max_structured_output_retries: int = Field(
        default=3,
    )
    max_concurrent_research_units: int = Field(
        default=5,
    )
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
    )
    max_researcher_iterations: int = Field(
        default=10,
    )
    max_react_tool_calls: int = Field(
        default=5,
    )
    max_content_length: int = Field(
        default=5000,
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True

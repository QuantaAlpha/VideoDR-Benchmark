"""
Prompt templates for LLM-based failure analysis
"""

# Failure Category Description
FAILURE_CATEGORY_DESCRIPTION = """
Failure Category Definitions:

1. **categorical_error** (Category/Name Recognition Error)
   - The model incorrectly identifies, classifies, or names entities
   - Examples: Wrong person/place/company names, incorrect category selection, confusing similar entities
   - Root cause: Over-reliance on single source, insufficient entity disambiguation, inadequate information verification

2. **numerical_error** (Numerical Calculation Error)
   - The model makes errors in numerical calculations, extraction, or comparisons
   - Examples: Arithmetic mistakes, unit conversion errors, wrong value extraction from charts/text, precision issues
   - Root cause: Calculation errors, unit conversion mistakes, chart interpretation errors, lack of result validation

3. **semantic_mismatch** (Semantic Mismatch)
   - The model's answer is semantically related but not correct compared to the standard answer
   - Examples: Using synonyms with slight meaning deviation, correct format but wrong content, understanding bias
   - Root cause: Insufficient precision in understanding requirements, semantic understanding deviation, failure to match standard answer's semantic range

4. **incomplete_research** (Incomplete Research)
   - The model fails to conduct sufficient research to obtain all information needed to answer the question
   - Examples: Too few research iterations, missing key research topics, premature stopping, failure to verify key information
   - Root cause: Premature conclusions, failure to identify all required information points, incomplete research strategy

5. **information_not_found** (Information Not Found)
   - The model cannot find or retrieve key information needed to answer the question
   - Examples: Research queries return no relevant information, information doesn't exist in sources, poor search strategy
   - Root cause: Inappropriate search keywords, source limitations, information genuinely unavailable, flawed search strategy

6. **reasoning_error** (Reasoning Error)
   - The model makes errors in logical reasoning, problem decomposition, or information synthesis
   - Examples: Broken logical reasoning chains, incorrect problem decomposition, information synthesis errors, wrong assumptions
   - Root cause: Insufficient logical reasoning ability, improper problem decomposition, flawed information synthesis strategy

7. **tool_usage_error** (Tool Usage Error)
   - The model makes errors when using tools (e.g., ConductResearch, think_tool)
   - Examples: Incorrect tool call format, improper tool usage sequence, wrong tool parameters
   - Root cause: Tool usage mistakes, parameter errors, sequence errors

**IMPORTANT**: 
- The above 7 categories are common failure types, but you are NOT limited to them
- If none of these categories accurately describe the failure, you may create a NEW, descriptive category name
- The new category name should be specific and meaningful (e.g., "format_error", "context_misunderstanding", "multi_step_failure")
- DO NOT use generic terms like "other" or "unknown" - always use a descriptive category name
- If a case seems to fit multiple categories, choose the PRIMARY root cause or create a new category that best captures it
"""

# System message for LLM
SYSTEM_MESSAGE = """You are an expert failure analysis specialist with deep expertise in analyzing AI model failures. Your task is to classify failure cases into appropriate failure categories based on root cause analysis. You may use common categories or create new descriptive category names if needed. Always use specific, meaningful category names - never use generic terms like 'other' or 'unknown'. Always provide clear, evidence-based reasoning and return responses in valid JSON format only."""

# System message for retry
SYSTEM_MESSAGE_RETRY = """You are an expert failure analysis specialist. Always use specific, descriptive category names. Never use 'other' or 'unknown'."""

# Main analysis prompt template
def get_analysis_prompt(question, real_answer, model_answer, trace_summary):
    """Generate the main analysis prompt"""
    return f"""You are an expert failure analysis specialist. Your task is to analyze why a video question-answering model failed and classify the failure into the appropriate category.

## Case Information

**Question**: {question}

**Standard Answer**: {real_answer}

**Model Answer**: {model_answer}

**Execution Trace Summary**:
{trace_summary}

## Context
This is a manually evaluated error case (human evaluator marked check=0, indicating the model's answer is incorrect). Your goal is to identify the PRIMARY root cause of the failure.

## Failure Category Definitions

{FAILURE_CATEGORY_DESCRIPTION}

## Analysis Process

Follow these steps to determine the failure category:

1. **Answer Comparison**: 
   - Compare the model's answer with the standard answer
   - Identify specific differences (wrong names/numbers, missing information, semantic deviations, etc.)

2. **Trace Analysis**:
   - Review the research topics the model explored
   - Examine the thinking process (reflections)
   - Assess tool usage (number of research calls, think calls)
   - Determine if the model conducted sufficient research

3. **Root Cause Identification**:
   - Determine the PRIMARY reason for failure
   - If multiple issues exist, identify the most fundamental one
   - Consider: Did the model fail because it couldn't find information, didn't research enough, made a calculation error, or misunderstood the question?

4. **Category Selection**:
   - First, check if the failure fits one of the 7 common categories above
   - If none of them accurately describe the failure, create a NEW, descriptive category name
   - The category name should be specific, meaningful, and clearly indicate the failure type
   - Use lowercase with underscores (e.g., "format_error", "context_misunderstanding", "multi_step_failure")
   - **CRITICAL**: DO NOT use generic terms like "other" or "unknown". Always use a descriptive, specific category name.

## Output Requirements

Return ONLY a valid JSON object (no markdown, no explanations, no additional text):

{{
    "failure_category": "<a descriptive category name. You may use one of the 7 common categories (categorical_error, numerical_error, semantic_mismatch, incomplete_research, information_not_found, reasoning_error, tool_usage_error) OR create a new descriptive category name if none fit. Use lowercase with underscores. DO NOT use 'other' or 'unknown'>",
    "reasoning": "<2-4 sentences explaining: (1) what went wrong, (2) why this category fits (or why a new category was created), (3) specific evidence from the trace/answers that supports this classification>",
    "key_issues": ["<concise issue 1>", "<concise issue 2>", "<concise issue 3>"]
}}

Important: Return ONLY the JSON object, nothing else."""

# Retry prompt template
def get_retry_prompt(question, real_answer, model_answer, trace_summary, failure_category):
    """Generate the retry prompt when LLM returns forbidden category"""
    return f"""The previous analysis returned a forbidden category '{failure_category}'. 

Please re-analyze this failure case and provide a SPECIFIC, DESCRIPTIVE category name.

**Question**: {question}
**Standard Answer**: {real_answer}
**Model Answer**: {model_answer}
**Trace Summary**: {trace_summary}

You may use one of these common categories: categorical_error, numerical_error, semantic_mismatch, incomplete_research, information_not_found, reasoning_error, tool_usage_error.

OR create a NEW descriptive category name if none fit (e.g., "format_error", "context_misunderstanding", "multi_step_failure").

DO NOT use "other" or "unknown". Return ONLY a JSON object with failure_category, reasoning, and key_issues."""


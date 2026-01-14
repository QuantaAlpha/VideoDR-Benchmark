#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Judge Prompt Templates

This module provides prompt templates for using LLM as a judge to evaluate
model answers against standard answers.
"""

SYSTEM_MESSAGE = """You are an expert answer evaluator with strong analytical capabilities. 
Your role is to carefully analyze and compare answers using a systematic evaluation framework. 
You should apply critical thinking to determine correctness based on semantic equivalence and factual accuracy, 
rather than relying solely on exact textual matching. Always provide clear, detailed reasoning for your judgments."""


def get_judge_prompt(question: str, standard_answer: str, model_answer: str) -> str:
    """
    Generate the prompt for judging if a model answer is correct.
    
    Args:
        question: The question being asked
        standard_answer: The correct/standard answer
        model_answer: The answer from the model to be evaluated
    
    Returns:
        The formatted prompt string
    """
    prompt = f"""You are an expert answer evaluator. Your task is to determine whether a model's answer is correct by comparing it with a standard answer.

Question: {question}

Standard Answer: {standard_answer}

Model Answer: {model_answer}

**Evaluation Framework:**

1. **Identify the answer type**: Determine whether the answer is numerical, categorical, descriptive, or a combination. Consider what type of information the question is asking for.

2. **Extract core information**: Identify the essential information in both answers, ignoring formatting, stylistic differences, and non-essential details.

3. **Semantic equivalence assessment**: 
   - For numerical answers: Check if the values are equivalent considering units, precision, and representation formats
   - For categorical answers: Check if they refer to the same entity or concept, accounting for synonyms and alternative names
   - For descriptive answers: Check if they convey the same meaning and key information

4. **Completeness check**: Determine if the model answer contains all critical information required by the question, or if missing information would make the answer incorrect.

5. **Contextual relevance**: Consider the question's intent and whether the model answer adequately addresses what was asked.

**Decision Principles:**
- Prioritize semantic correctness over exact textual matching
- Allow reasonable variations in format, units, and expression
- Be strict about factual accuracy and completeness of essential information
- Distinguish between related but distinct entities or concepts
- Consider the question's context and requirements when making judgments

**Output Format:**
Provide your judgment in JSON format:
{{
    "is_correct": true/false,
    "reasoning": "Your detailed reasoning explaining: (1) the answer type identified, (2) how you compared the answers, (3) what differences you found (if any), and (4) why you concluded the answer is correct or incorrect (in English)"
}}
"""
    return prompt


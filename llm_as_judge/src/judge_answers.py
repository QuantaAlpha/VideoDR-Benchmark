#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM as Judge Analysis Tool

This module provides functionality to evaluate model answers using LLM as a judge.
It loads all data from predictions.json file and uses an LLM to determine if model
answers are correct compared to ground truth answers.

Usage:
    python judge_answers.py --workers 5 --predictions data/predictions.json
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# Import from local module
try:
    from .judge_prompts import SYSTEM_MESSAGE, get_judge_prompt
except ImportError:
    # Fallback for direct script execution
    from judge_prompts import SYSTEM_MESSAGE, get_judge_prompt

# Thread-safe file write lock
file_lock = threading.Lock()


class TeeOutput:
    """
    Context manager that redirects output to both console and file.
    
    This class allows simultaneous output to stdout/stderr and a log file,
    useful for long-running analysis tasks.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize TeeOutput with a file path.
        
        Args:
            file_path: Path to the log file
        """
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def write(self, text: str) -> None:
        """Write text to both stdout and file."""
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
        
    def flush(self) -> None:
        """Flush both stdout and file."""
        self.stdout.flush()
        self.file.flush()
        
    def close(self) -> None:
        """Close the file handle."""
        if self.file:
            self.file.close()
            
    def __enter__(self):
        """Enter context manager, redirect stdout/stderr."""
        sys.stdout = self
        sys.stderr = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, restore stdout/stderr."""
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        if self.file:
            self.file.close()


def load_env_file() -> None:
    """
    Load environment variables from .env file.
    
    Tries to find .env file in the parent directory (llm_as_judge/.env)
    or current working directory. Uses python-dotenv if available,
    otherwise manually parses the file.
    """
    current_file = Path(__file__).resolve()
    env_path = current_file.parent.parent / '.env'
    
    if not env_path.exists():
        env_path = Path('.env')
    
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            # Manual parsing if dotenv is not installed
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value


# Load environment variables at module import (only if .env exists)
try:
    load_env_file()
except Exception:
    # Ignore errors if .env file doesn't exist or can't be read
    pass


def create_client(base_url: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
    """
    Create an OpenAI client instance.
    
    Args:
        base_url: API base URL (defaults to LLM_BASE_URL env var)
        api_key: API key (defaults to LLM_API_KEY env var)
        
    Returns:
        OpenAI client instance
        
    Raises:
        ValueError: If API key is not provided
    """
    if base_url is None:
        base_url = os.getenv('LLM_BASE_URL', '')
    if api_key is None:
        api_key = os.getenv('LLM_API_KEY', '')
    
    if not api_key:
        raise ValueError(
            "API key is required. Set LLM_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    return OpenAI(base_url=base_url, api_key=api_key)


def judge_answer(
    question: str,
    standard_answer: str,
    model_answer: str,
    client: Optional[OpenAI] = None
) -> Tuple[bool, str]:
    """
    Use LLM to judge if a model answer is correct.
    
    Args:
        question: The question being asked
        standard_answer: The correct/standard answer
        model_answer: The model's answer to evaluate
        client: OpenAI client instance (creates new one if None)
        
    Returns:
        Tuple of (is_correct: bool, reasoning: str)
    """
    if pd.isna(model_answer) or not model_answer or not str(model_answer).strip():
        return False, "Answer is empty"
    
    # Create client if not provided
    if client is None:
        client = create_client()
    
    # Get prompt from judge_prompts module
    prompt = get_judge_prompt(question, standard_answer, model_answer)
    
    try:
        response = client.chat.completions.create(
            model="/model/DeepSeek-V3",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        try:
            # Extract JSON part
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            is_correct = result.get("is_correct", False)
            reasoning = result.get("reasoning", "No reasoning provided")
            return is_correct, reasoning
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract from text
            result_lower = result_text.lower()
            if "true" in result_lower or "correct" in result_lower:
                return True, result_text
            elif "false" in result_lower or "incorrect" in result_lower:
                return False, result_text
            else:
                return False, f"Failed to parse judgment result: {result_text}"
    
    except Exception as e:
        print(f"LLM judgment error: {e}")
        return False, f"Judgment process error: {str(e)}"


def save_result_append(result: Dict, csv_path: str = "judge_results.csv") -> None:
    """
    Thread-safe function to append a single result to CSV file.
    
    If the file doesn't exist, creates a new file with headers.
    Uses file locking to ensure thread safety in concurrent execution.
    
    Args:
        result: Dictionary containing result data to save
        csv_path: Path to the CSV file
    """
    with file_lock:
        file_exists = os.path.exists(csv_path)
        
        # Convert to DataFrame
        df_new = pd.DataFrame([result])
        
        try:
            # Append mode write to CSV
            if file_exists:
                # Append mode, no header
                df_new.to_csv(
                    csv_path,
                    mode='a',
                    header=False,
                    index=False,
                    encoding='utf-8-sig'
                )
            else:
                # New file, write header
                df_new.to_csv(
                    csv_path,
                    mode='w',
                    header=True,
                    index=False,
                    encoding='utf-8-sig'
                )
        except Exception as e:
            # If append fails, try to rewrite entire file (fallback)
            print(f"\nWarning: Append save failed ({e}), attempting rewrite...")
            try:
                # Read existing data
                if file_exists:
                    existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
                    combined_df = pd.concat([existing_df, df_new], ignore_index=True)
                else:
                    combined_df = df_new
                combined_df.to_csv(
                    csv_path,
                    mode='w',
                    header=True,
                    index=False,
                    encoding='utf-8-sig'
                )
            except Exception as e2:
                print(f"Error: Save failed: {e2}")
                raise


def process_single_task(task_info: Dict, csv_path: str) -> Dict:
    """
    Process a single task (question + model combination).
    
    Args:
        task_info: Dictionary containing task information:
            - question_id: Question ID
            - question: Question text
            - standard_answer: Standard answer (ground truth)
            - model_identifier: Model identifier (e.g., "gpt-4o-key-frame.workflow" or "qwen")
            - model_answer: Model's answer text
        csv_path: Path to CSV file for saving results
        
    Returns:
        Dictionary containing evaluation result
    """
    question_id = task_info['question_id']
    question = task_info['question']
    standard_answer = task_info['standard_answer']
    model_identifier = task_info['model_identifier']
    model_answer = task_info.get('model_answer', '')
    
    # Create independent client
    client = create_client()
    
    # Use LLM to judge
    try:
        is_correct, reasoning = judge_answer(question, standard_answer, model_answer, client)
    except Exception as e:
        print(f"\nError judging question {question_id} model {model_identifier}: {e}")
        is_correct, reasoning = False, f"Judgment error: {str(e)}"
    
    # Build result record
    result = {
        "question_id": question_id,
        "question": question,
        "standard_answer": standard_answer,
        "model": model_identifier,
        "model_answer": model_answer if model_answer else "",
        "is_correct": is_correct,
        "reasoning": reasoning
    }
    
    # Save to CSV file in real-time
    try:
        save_result_append(result, csv_path)
    except Exception as e:
        print(f"\nFailed to save result: {e}")
    
    return result


def load_predictions(predictions_file: str) -> Dict:
    """
    Load predictions from JSON file.
    
    Expected structure:
    {
      "question_id": {
        "question": "...",
        "ground_truth": "...",
        "model_name": {
          "model_name_1": {
            "agentic": {"answer": "..."},
            "workflow": {"answer": "..."}
          }
        }
      }
    }
    
    Args:
        predictions_file: Path to predictions.json file (relative or absolute)
        
    Returns:
        Dictionary containing all predictions data
        
    Raises:
        FileNotFoundError: If predictions file doesn't exist
        ValueError: If predictions file is invalid
    """
    # If relative path, try to resolve it relative to script location
    if not os.path.isabs(predictions_file):
        # Try current working directory first
        if os.path.exists(predictions_file):
            pass  # Use as is
        else:
            # Try relative to script location
            script_dir = Path(__file__).parent.parent  # llm_as_judge directory
            alt_path = script_dir / predictions_file
            if alt_path.exists():
                predictions_file = str(alt_path)
            else:
                # Try relative to script's src directory
                src_alt_path = Path(__file__).parent / predictions_file
                if src_alt_path.exists():
                    predictions_file = str(src_alt_path)
    
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_file}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Script location: {Path(__file__).parent}"
        )
    
    try:
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        return predictions
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in predictions file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading predictions file: {e}")


def analyze_all_questions(
    resume: bool = True,
    max_workers: int = 5,
    predictions_file: str = "data/predictions.json",
    csv_path: str = "judge_results.csv"
) -> pd.DataFrame:
    """
    Analyze all questions using concurrent processing.
    
    Reads all data (questions, ground truth, and model answers) from predictions.json.
    
    Args:
        resume: Whether to resume from existing results
        max_workers: Maximum number of concurrent threads
        predictions_file: Path to predictions.json file
        csv_path: Output CSV file path
        
    Returns:
        DataFrame containing all evaluation results
        
    Raises:
        FileNotFoundError: If predictions file cannot be found
        ValueError: If predictions file is invalid or empty
    """
    # Load predictions from JSON file
    predictions = load_predictions(predictions_file)
    
    if not predictions:
        raise ValueError(f"Predictions file is empty: {predictions_file}")
    
    print(f"Loaded predictions from {predictions_file}")
    print(f"Found {len(predictions)} questions")
    print(f"Using {max_workers} concurrent workers")
    
    # Result file path
    excel_path_final = csv_path.replace('.csv', '.xlsx')
    
    # Prepare result storage
    processed_keys: Set[Tuple[int, str]] = set()
    
    # If resume is enabled, try to load existing results
    if resume and os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
            processed_keys = {
                (int(r['question_id']), r['model'])
                for _, r in existing_df.iterrows()
            }
            print(f"Loaded {len(existing_df)} existing results, continuing with remaining questions")
        except Exception as e:
            print(f"Failed to load existing results: {e}, starting fresh")
            # If load fails, backup the problematic file
            if os.path.exists(csv_path):
                backup_path = f"{csv_path}.backup_{int(time.time())}"
                os.rename(csv_path, backup_path)
                print(f"Backed up original file to: {backup_path}")
    
    # Prepare all tasks
    tasks = []
    
    # Count total model/mode combinations for statistics
    total_combinations = 0
    
    # Iterate through all questions in predictions
    for question_id_str, question_data in predictions.items():
        try:
            question_id = int(question_id_str)
        except ValueError:
            print(f"Warning: Invalid question_id '{question_id_str}', skipping")
            continue
        
        question = question_data.get('question', '')
        ground_truth = question_data.get('ground_truth', '')
        model_name_dict = question_data.get('model_name', {})
        
        if not question or not ground_truth:
            print(f"Warning: Question {question_id} missing question or ground_truth, skipping")
            continue
        
        # Iterate through all models
        for model_name, model_data in model_name_dict.items():
            # Process agentic and workflow modes
            for mode in ['agentic', 'workflow']:
                if mode not in model_data:
                    continue
                
                mode_data = model_data[mode]
                answer = mode_data.get('answer', '')
                
                # Construct model identifier with mode suffix
                # Both agentic and workflow should have suffix for consistency
                model_identifier = f"{model_name}.{mode}"
                
                key = (question_id, model_identifier)
                if key not in processed_keys:
                    tasks.append({
                        'question_id': question_id,
                        'question': question,
                        'standard_answer': ground_truth,
                        'model_identifier': model_identifier,
                        'model_answer': answer
                    })
                    total_combinations += 1
    
    print(f"Found {total_combinations} model/mode combinations to evaluate")
    print(f"Pending tasks: {len(tasks)}")
    
    total_tasks = total_combinations
    completed = len(processed_keys)
    
    # Use thread pool for concurrent processing
    results = []
    with tqdm(total=total_tasks, initial=completed, desc="Analysis progress") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_single_task, task, csv_path): task
                for task in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    pbar.set_postfix({
                        "status": f"Q{result['question_id']}-{result['model']}"
                    })
                except Exception as e:
                    print(f"\nError processing task Q{task['question_id']}-{task['model_identifier']}: {e}")
                    # Record error even if processing fails
                    result = {
                        "question_id": task['question_id'],
                        "question": task['question'],
                        "standard_answer": task['standard_answer'],
                        "model": task['model_identifier'],
                        "model_answer": task.get('model_answer', ''),
                        "is_correct": False,
                        "reasoning": f"Processing error: {str(e)}"
                    }
                    results.append(result)
                    save_result_append(result, csv_path)
                finally:
                    pbar.update(1)
    
    # Re-read CSV file to ensure data completeness, then save as Excel
    try:
        final_df = pd.read_csv(csv_path, encoding='utf-8-sig')
        final_df.to_excel(excel_path_final, index=False)
        print(f"\nFinal results saved to {excel_path_final}")
    except Exception as e:
        print(f"\nError saving Excel file: {e}, but CSV file is saved")
        # If read fails, use in-memory data
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_excel(excel_path_final, index=False)
    
    # Generate statistics report
    try:
        stats_df = pd.read_csv(csv_path, encoding='utf-8-sig')
        generate_statistics_report(stats_df)
    except Exception as e:
        print(f"Error generating statistics report: {e}")
        if results:
            generate_statistics_report(pd.DataFrame(results))
    
    print(f"\nAnalysis complete! Results saved to {csv_path}")
    
    return pd.read_csv(csv_path, encoding='utf-8-sig') if os.path.exists(csv_path) else pd.DataFrame(results)


def generate_statistics_report(results_df: pd.DataFrame) -> None:
    """
    Generate a statistics report from evaluation results.
    
    Args:
        results_df: DataFrame containing evaluation results
    """
    report = []
    report.append("=" * 80)
    report.append("LLM as Judge Analysis Report")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    total_questions = results_df['question_id'].nunique()
    total_evaluations = len(results_df)
    report.append(f"Total questions: {total_questions}")
    report.append(f"Total evaluations: {total_evaluations}")
    report.append("")
    
    # Statistics by model
    report.append("-" * 80)
    report.append("Accuracy Statistics by Model")
    report.append("-" * 80)
    
    # Calculate accuracy with denominator of 100 (total questions)
    total_questions = 100
    
    model_stats_list = []
    for model_name, group in results_df.groupby('model'):
        total_evaluated = len(group)
        correct_count = group['is_correct'].sum()
        accuracy = (correct_count / total_questions) * 100
        model_stats_list.append({
            'Model': model_name,
            'Total': total_evaluated,
            'Correct': correct_count,
            'Accuracy(%)': round(accuracy, 2)
        })
    
    model_stats_df = pd.DataFrame(model_stats_list)
    model_stats_df = model_stats_df.sort_values('Model')
    report.append(model_stats_df.to_string(index=False))
    report.append("")
    
    # Detailed error analysis
    report.append("-" * 80)
    report.append("Incorrect Answer Examples (Top 5)")
    report.append("-" * 80)
    
    wrong_answers = results_df[results_df['is_correct'] == False].head(5)
    for _, row in wrong_answers.iterrows():
        report.append(f"\nQuestion ID: {row['question_id']}")
        report.append(f"Model: {row['model']}")
        report.append(f"Question: {row['question']}")
        report.append(f"Standard Answer: {row['standard_answer']}")
        report.append(f"Model Answer: {row['model_answer']}")
        report.append(f"Reasoning: {row['reasoning']}")
        report.append("-" * 80)
    
    # Save report
    report_text = "\n".join(report)
    with open("judge_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print("\n" + report_text)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='LLM as Judge Analysis Tool - Evaluate models from predictions.json'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of concurrent threads (default: 5)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_false',
        dest='resume',
        help='Disable resume functionality, start from scratch'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        default='data/predictions.json',
        help='Path to predictions.json file (default: data/predictions.json)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log file path (default: judge_results_log.txt)'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default='judge_results.csv',
        help='Output CSV file path (default: judge_results.csv)'
    )
    
    args = parser.parse_args()
    
    # Set log file path
    if args.log_file is None:
        log_file = args.output_csv.replace('.csv', '_log.txt')
    else:
        log_file = args.log_file
    
    # Validate worker count
    if args.workers < 1:
        print("Warning: Worker count must be greater than 0, using default value 5")
        args.workers = 5
    elif args.workers > 20:
        print("Warning: Worker count too high may cause API rate limiting, recommend <= 20")
    
    # Create output redirection, output to both console and file
    with TeeOutput(log_file):
        print("=" * 80)
        print("LLM as Judge Analysis Tool")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {log_file}")
        print(f"Concurrent workers: {args.workers}")
        print(f"Resume mode: {'Enabled' if args.resume else 'Disabled'}")
        print("=" * 80)
        
        print("\nStarting LLM as Judge analysis...")
        try:
            results_df = analyze_all_questions(
                resume=args.resume,
                max_workers=args.workers,
                predictions_file=args.predictions,
                csv_path=args.output_csv
            )
            print("\n" + "=" * 80)
            print(f"Analysis complete! End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
        except Exception as e:
            print(f"\nError during analysis: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()


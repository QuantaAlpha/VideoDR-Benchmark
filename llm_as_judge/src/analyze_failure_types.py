#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze failure traces error types using LLM based on manual evaluation results

Usage:
    python analyze_failure_types.py --max_workers 4
"""

import json
import os
import pandas as pd
import numpy as np
import re
import argparse
from pathlib import Path
from collections import defaultdict
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    # Try to find .env file in parent directory (llm_as_judge/.env)
    current_file = Path(__file__).resolve()
    env_path = current_file.parent.parent / '.env'
    if not env_path.exists():
        # Try current working directory
        env_path = Path('.env')
    
    if env_path.exists():
        try:
            # Try using python-dotenv if available
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
                        # Only set if not already in environment
                        if key and key not in os.environ:
                            os.environ[key] = value

# Load .env file at module import
load_env_file()

from prompts import (
    FAILURE_CATEGORY_DESCRIPTION,
    SYSTEM_MESSAGE,
    SYSTEM_MESSAGE_RETRY,
    get_analysis_prompt,
    get_retry_prompt
)

# Model name mapping
MODEL_NAME_MAP = {
    'qwen': 'Qwen:Qwen3-Omni-30B-A3B-Instruct',
    'internvl': 'OpenGVLab:InternVL3_5-14B',
    'minicpm': 'OpenBMB:MiniCPM-V-4.5',
    'gpt4o': 'gpt-4o-key-frame',
    'gemini': 'gemini-3-pro-preview-key_frame',
    'gpt52': 'gpt-5.2-key_frame'
}

# Initialize OpenAI client
def create_client(base_url=None, api_key=None):
    """Create OpenAI client"""
    if base_url is None:
        base_url = os.getenv('LLM_BASE_URL',"")
    if api_key is None:
        api_key = os.getenv('LLM_API_KEY', "")
    
    if not api_key:
        raise ValueError("API key is required. Set LLM_API_KEY environment variable or pass api_key parameter.")
    
    return OpenAI(
        base_url=base_url,
        api_key=api_key
    )

def load_manual_evaluation(excel_file):
    """Load manual evaluation results"""
    # Read Excel file, first row is empty, second row contains column names
    df_raw = pd.read_excel(excel_file, header=None)
    
    # Use second row as column names
    df = pd.read_excel(excel_file, header=1)
    
    # Get actual column names
    actual_cols = list(df.columns)
    
    # Build column mapping
    col_mapping = {}
    
    # Basic columns (first 5): id, url, question, label, Real Answer
    basic_cols = ['id', 'url', 'question', 'label', 'real_answer']
    for i, col_name in enumerate(basic_cols):
        if i < len(actual_cols):
            col_mapping[actual_cols[i]] = col_name
    
    # Standard workflow columns (indices 5-14): qwen, qwen-check, internvl, internvl-check, etc.
    workflow_models = ['qwen', 'internvl', 'minicpm', 'gpt4o', 'gemini']
    for i, model in enumerate(workflow_models):
        model_idx = 5 + i * 2
        check_idx = 5 + i * 2 + 1
        if model_idx < len(actual_cols):
            col_mapping[actual_cols[model_idx]] = f'workflow_{model}'
        if check_idx < len(actual_cols):
            col_mapping[actual_cols[check_idx]] = f'workflow_{model}_check'
    
    # Standard agentic columns (indices 15-24): qwen.1, qwen-check.1, etc.
    for i, model in enumerate(workflow_models):
        model_idx = 15 + i * 2
        check_idx = 15 + i * 2 + 1
        if model_idx < len(actual_cols):
            col_mapping[actual_cols[model_idx]] = f'agentic_{model}'
        if check_idx < len(actual_cols):
            col_mapping[actual_cols[check_idx]] = f'agentic_{model}_check'
    
    # Handle gpt-5.2 columns (indices 26-29)
    # Index 26: 'gpt-5.2' (agentic)
    # Index 27: 'Unnamed: 27' (agentic check)
    # Index 28: 'gpt-5.2.1' (workflow)
    # Index 29: 'Unnamed: 29' (workflow check)
    if len(actual_cols) > 26 and 'gpt-5.2' in str(actual_cols[26]):
        col_mapping[actual_cols[26]] = 'agentic_gpt52'
        if len(actual_cols) > 27:
            col_mapping[actual_cols[27]] = 'agentic_gpt52_check'
    if len(actual_cols) > 28 and 'gpt-5.2' in str(actual_cols[28]):
        col_mapping[actual_cols[28]] = 'workflow_gpt52'
        if len(actual_cols) > 29:
            col_mapping[actual_cols[29]] = 'workflow_gpt52_check'
    
    # Rename columns
    df = df.rename(columns=col_mapping)
    
    # Clean data: remove empty rows, convert check columns to numeric
    df = df[df['id'].notna()].copy()
    
    # Convert check columns to numeric (1=correct, 0=incorrect)
    for col in df.columns:
        if 'check' in col:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    return df

def load_trace_file(trace_dir, model_name, workflow_type, question_id):
    """Load trace file"""
    trace_path = trace_dir / model_name / workflow_type / f"{question_id}.json"
    if not trace_path.exists():
        return None
    
    try:
        with open(trace_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read trace file {trace_path}: {e}")
        return None

def extract_trace_summary(trace_data):
    """Extract key information summary from trace"""
    if trace_data is None:
        return "Trace file does not exist"
    
    messages = trace_data.get('values', {}).get('messages', [])
    
    # Count tool calls
    think_count = 0
    research_count = 0
    research_topics = []
    think_reflections = []
    final_answer = trace_data.get('answer', '')
    duration = trace_data.get('duration_seconds', 0)
    
    for message in messages:
        if message.get('type') == 'ai' and 'tool_calls' in message:
            for tool_call in message.get('tool_calls', []):
                tool_name = tool_call.get('name', '')
                if tool_name == 'think_tool':
                    think_count += 1
                    reflection = tool_call.get('args', {}).get('reflection', '')
                    if reflection:
                        think_reflections.append(reflection[:200])  # Take first 200 characters
                elif tool_name == 'ConductResearch':
                    research_count += 1
                    topic = tool_call.get('args', {}).get('research_topic', '')
                    if topic:
                        research_topics.append(topic[:200])  # Take first 200 characters
    
    summary = f"""Trace Statistics:
- think_tool calls: {think_count}
- ConductResearch calls: {research_count}
- Duration: {duration:.2f} seconds
- Final Answer: {final_answer[:500] if final_answer else 'None'}

Research Topics: {', '.join(research_topics[:5]) if research_topics else 'None'}

Thinking Process (first 3 reflections):
{chr(10).join(think_reflections[:3]) if think_reflections else 'None'}
"""
    return summary

def force_classify_failure(question, real_answer, model_answer, trace_summary):
    """
    Force classification when LLM returns 'other' or 'unknown'
    Uses rule-based logic to assign to one of the 7 specific categories
    """
    
    # 1. Check for numerical errors
    if isinstance(real_answer, (int, float)) or (isinstance(real_answer, str) and re.match(r'^-?\d+\.?\d*$', str(real_answer).replace(',', '').strip())):
        if isinstance(model_answer, (int, float)) or (isinstance(model_answer, str) and re.match(r'^-?\d+\.?\d*$', str(model_answer).replace(',', '').strip())):
            try:
                real_val = float(str(real_answer).replace(',', ''))
                model_val = float(str(model_answer).replace(',', ''))
                if abs(real_val - model_val) > 0.01:
                    return 'numerical_error'
            except:
                pass
    
    # 2. Check for empty or missing answers
    if pd.isna(model_answer) or str(model_answer).strip() == '' or str(model_answer).strip().lower() in ['[空答案]', '[empty_answer]', 'nan', 'none', 'missing_data']:
        return 'information_not_found'
    
    # 3. Check trace summary for research completeness
    if 'research_count: 0' in trace_summary or 'ConductResearch calls: 0' in trace_summary:
        return 'incomplete_research'
    
    # Extract research count from trace summary
    research_match = re.search(r'ConductResearch calls?:\s*(\d+)', trace_summary)
    if research_match:
        research_count = int(research_match.group(1))
        if research_count == 0:
            return 'incomplete_research'
        elif research_count == 1:
            # Check think count
            think_match = re.search(r'think_tool calls?:\s*(\d+)', trace_summary)
            if think_match and int(think_match.group(1)) <= 1:
                return 'incomplete_research'
    
    # 4. Check for semantic similarity (partial match)
    real_str = str(real_answer).lower().strip()
    model_str = str(model_answer).lower().strip()
    
    if real_str in model_str or model_str in real_str:
        return 'semantic_mismatch'
    
    # 5. Check for categorical errors (completely different)
    real_words = set(re.findall(r'\w+', real_str))
    model_words = set(re.findall(r'\w+', model_str))
    common_words = real_words & model_words
    
    if len(real_words) > 0:
        overlap_ratio = len(common_words) / len(real_words)
        if overlap_ratio < 0.3:
            return 'categorical_error'
    
    # 6. Default to categorical_error if nothing else fits
    return 'categorical_error'

def llm_analyze_failure_type(question, real_answer, model_answer, trace_summary, client):
    """Use LLM to analyze failure type"""
    
    prompt = get_analysis_prompt(question, real_answer, model_answer, trace_summary)

    try:
        response = client.chat.completions.create(
            model="/model/DeepSeek-V3",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON
        try:
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            failure_category = result.get('failure_category', '').lower().strip()
            
            # Only check if forbidden generic categories are used, allow any other descriptive categories
            if failure_category in ['other', 'unknown', '']:
                print(f"Warning: LLM returned forbidden generic category '{failure_category}'. Requesting re-analysis...")
                # Retry LLM call, emphasize not to use other/unknown
                retry_prompt = get_retry_prompt(question, real_answer, model_answer, trace_summary, failure_category)
                
                try:
                    retry_response = client.chat.completions.create(
                        model="/model/DeepSeek-V3",
                        messages=[
                            {"role": "system", "content": SYSTEM_MESSAGE_RETRY},
                            {"role": "user", "content": retry_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=1500
                    )
                    retry_text = retry_response.choices[0].message.content.strip()
                    if "```json" in retry_text:
                        retry_text = retry_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in retry_text:
                        retry_text = retry_text.split("```")[1].split("```")[0].strip()
                    result = json.loads(retry_text)
                    failure_category = result.get('failure_category', '').lower().strip()
                    if failure_category in ['other', 'unknown', '']:
                        print(f"Error: Retry also returned forbidden category. Using fallback classification.")
                        failure_category = force_classify_failure(question, real_answer, model_answer, trace_summary)
                        result['failure_category'] = failure_category
                        result['reasoning'] = f"[Fallback after retry failed] " + result.get('reasoning', '')
                except Exception as retry_e:
                    print(f"Retry failed: {retry_e}. Using fallback classification.")
                    failure_category = force_classify_failure(question, real_answer, model_answer, trace_summary)
                    result['failure_category'] = failure_category
                    result['reasoning'] = f"[Fallback after retry exception] " + result.get('reasoning', '')
            
            return result
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"LLM returned text:\n{result_text[:500]}")
            # Only use fallback if absolutely necessary
            forced_category = force_classify_failure(question, real_answer, model_answer, trace_summary)
            return {
                'failure_category': forced_category,
                'reasoning': f'JSON parsing failed, used fallback classification: {str(e)}',
                'key_issues': []
            }
    except Exception as e:
        print(f"LLM analysis failed: {e}")
        # Only use fallback if absolutely necessary
        forced_category = force_classify_failure(question, real_answer, model_answer, trace_summary)
        return {
            'failure_category': forced_category,
            'reasoning': f'LLM analysis failed, used fallback classification: {str(e)}',
            'key_issues': []
        }

def count_tool_calls(messages):
    """Count think_tool and ConductResearch calls"""
    think_count = 0
    research_count = 0
    
    for message in messages:
        if message.get('type') == 'ai' and 'tool_calls' in message:
            for tool_call in message.get('tool_calls', []):
                tool_name = tool_call.get('name', '')
                if tool_name == 'think_tool':
                    think_count += 1
                elif tool_name == 'ConductResearch':
                    research_count += 1
    
    return think_count, research_count

def analyze_single_failure(args):
    """Analyze a single failure case (for concurrent processing)"""
    question_id, question, real_answer, model_key, model_name, workflow_type, model_answer, trace_dir, client = args
    
    try:
        # Load trace file
        trace_data = load_trace_file(trace_dir, model_name, workflow_type, question_id)
        
        # Extract trace summary
        trace_summary = extract_trace_summary(trace_data)
        
        # Use LLM to analyze failure type
        llm_result = llm_analyze_failure_type(
            question, real_answer, model_answer, trace_summary, client
        )
        
        # Extract trace statistics
        think_count = 0
        research_count = 0
        duration = 0
        if trace_data:
            messages = trace_data.get('values', {}).get('messages', [])
            think_count, research_count = count_tool_calls(messages)
            duration = trace_data.get('duration_seconds', 0)
        
        return {
            'question_id': question_id,
            'model': f'{model_key}-{workflow_type}',
            'question': question,
            'real_answer': real_answer,
            'model_answer': model_answer,
            'failure_category': llm_result.get('failure_category', 'other'),
            'failure_reasoning': llm_result.get('reasoning', ''),
            'key_issues': '; '.join(llm_result.get('key_issues', [])),
            'think_count': think_count,
            'research_count': research_count,
            'duration_seconds': duration,
            'trace_file': f'{model_name}/{workflow_type}/{question_id}.json' if trace_data else None
        }
    except Exception as e:
        print(f"\nError analyzing {model_key}-{workflow_type}, Question ID={question_id}: {e}")
        return None

def collect_failure_traces(eval_df, trace_dir, client, max_workers=4, target_models=None):
    """Collect all failure traces and analyze with LLM (supports concurrency)
    
    Args:
        eval_df: Evaluation dataframe
        trace_dir: Directory containing trace files
        client: LLM client
        max_workers: Number of concurrent workers
        target_models: List of model keys to analyze (e.g., ['gpt52', 'gpt4o']). 
                     If None, analyzes all models in MODEL_NAME_MAP.
    """
    failures = []
    tasks = []
    
    # If target_models is None, analyze all models
    if target_models is None:
        target_models = list(MODEL_NAME_MAP.keys())
    
    # Validate target_models
    invalid_models = [m for m in target_models if m not in MODEL_NAME_MAP]
    if invalid_models:
        print(f"Warning: Invalid model keys {invalid_models}. Available models: {list(MODEL_NAME_MAP.keys())}")
        target_models = [m for m in target_models if m in MODEL_NAME_MAP]
    
    if not target_models:
        print("Error: No valid models to analyze.")
        return pd.DataFrame()
    
    print(f"Target models to analyze: {target_models}")
    print(f"Collecting failure cases from {len(eval_df)} evaluation records...")
    
    for idx, row in eval_df.iterrows():
        question_id = int(row['id'])
        question = row['question']
        real_answer = row['real_answer']
        
        # Check each model and each workflow type
        for model_key, model_name in MODEL_NAME_MAP.items():
            # Only analyze specified models
            if model_key not in target_models:
                continue
                
            for workflow_type in ['workflow', 'agentic']:
                check_col = f'{workflow_type}_{model_key}_check'
                answer_col = f'{workflow_type}_{model_key}'
                
                if check_col not in row.index or answer_col not in row.index:
                    continue
                
                check_result = row[check_col]
                model_answer = row[answer_col]
                
                # check=0 means error
                if pd.notna(check_result) and int(check_result) == 0:
                    tasks.append((
                        question_id, question, real_answer, model_key, model_name, 
                        workflow_type, model_answer, trace_dir, client
                    ))
    
    print(f"Found {len(tasks)} failure cases to analyze")
    print(f"Using {max_workers} concurrent workers...")
    print()
    
    # Use thread pool for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(analyze_single_failure, task): task 
            for task in tasks
        }
        
        # Use tqdm to show progress
        with tqdm(total=len(tasks), desc="Analyzing failures", unit="case") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                question_id, _, _, model_key, _, workflow_type, _, _, _ = task
                
                try:
                    result = future.result()
                    if result is not None:
                        failures.append(result)
                        pbar.set_postfix({
                            'current': f"{model_key}-{workflow_type} Q{question_id}",
                            'completed': len(failures)
                        })
                except Exception as e:
                    print(f"\nError processing {model_key}-{workflow_type}, Question ID={question_id}: {e}")
                finally:
                    pbar.update(1)
    
    return pd.DataFrame(failures)

def generate_statistics(df_failures):
    """Generate statistics report"""
    print("=" * 100)
    print("Failure Trace Error Type Analysis Statistics (Based on LLM Analysis)")
    print("=" * 100)
    print()
    
    print(f"Total failure traces: {len(df_failures)}")
    print()
    
    # Statistics by model
    print("=" * 100)
    print("Statistics by Model")
    print("=" * 100)
    print()
    
    for model in sorted(df_failures['model'].unique()):
        model_failures = df_failures[df_failures['model'] == model]
        print(f"\n[Model] {model}")
        print(f"Failure traces: {len(model_failures)}")
        
        # Error type distribution
        if 'failure_category' in model_failures.columns:
            category_counts = model_failures['failure_category'].value_counts()
            print("\nError type distribution:")
            for category, count in category_counts.items():
                percentage = (count / len(model_failures)) * 100
                print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Average statistics
        print("\nAverage statistics:")
        print(f"  Average think count: {model_failures['think_count'].mean():.2f}")
        print(f"  Average research count: {model_failures['research_count'].mean():.2f}")
        print(f"  Average duration: {model_failures['duration_seconds'].mean():.2f} seconds")
    
    # Overall error type distribution
    print("\n" + "=" * 100)
    print("Overall Error Type Distribution")
    print("=" * 100)
    print()
    
    if 'failure_category' in df_failures.columns:
        overall_category_counts = df_failures['failure_category'].value_counts()
        for category, count in overall_category_counts.items():
            percentage = (count / len(df_failures)) * 100
            print(f"{category}: {count} ({percentage:.1f}%)")

def main():
    import sys
    import os
    from datetime import datetime
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze failure types using LLM')
    parser.add_argument('--max_workers', type=int, default=4, 
                       help='Number of concurrent workers (default: 4)')
    parser.add_argument('--excel_file', type=str, default='Video-LLM.xlsx',
                       help='Path to evaluation Excel file (default: Video-LLM.xlsx)')
    parser.add_argument('--trace_dir', type=str, default='traces',
                       help='Path to trace directory (default: traces)')
    parser.add_argument('--output_file', type=str, default='failure_analysis_llm.xlsx',
                       help='Output Excel file path (default: failure_analysis_llm.xlsx)')
    parser.add_argument('--log_file', type=str, default='failure_analysis_llm_log.txt',
                       help='Log file path (default: failure_analysis_llm_log.txt)')
    parser.add_argument('--base_url', type=str, default=None,
                       help='LLM API base URL (or set LLM_BASE_URL env var)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='LLM API key (or set LLM_API_KEY env var)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Model keys to analyze (e.g., --models gpt52 gpt4o). Available: qwen, internvl, minicpm, gpt4o, gemini, gpt52. If not specified, analyzes all models.')
    args = parser.parse_args()
    
    # File paths
    excel_file = args.excel_file
    trace_dir = Path(args.trace_dir)
    output_file = args.output_file
    log_file = args.log_file
    
    # Setup log file, output to both console and file
    log_path = Path(log_file)
    
    class Tee:
        """Output to both console and file"""
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Open log file
    log_file_handle = open(log_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file_handle)
    
    try:
        print("=" * 100)
        print(f"Failure Analysis Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)
        print()
        
        print("Initializing LLM client...")
        client = create_client(base_url=args.base_url, api_key=args.api_key)
        
        print("Loading manual evaluation results...")
        eval_df = load_manual_evaluation(excel_file)
        print(f"Loaded {len(eval_df)} evaluation records")
        
        print("\nCollecting failure traces and analyzing with LLM...")
        print("Note: This may take a while, please be patient...")
        df_failures = collect_failure_traces(eval_df, trace_dir, client, 
                                             max_workers=args.max_workers,
                                             target_models=args.models)
        print(f"\nCollected and analyzed {len(df_failures)} failure traces")
        
        print("\nGenerating statistics report...")
        generate_statistics(df_failures)
        
        # Save results
        print(f"\nSaving results to {output_file}...")
        output_path = Path(output_file)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_failures.to_excel(writer, sheet_name='All Failure Traces', index=False)
            
            # Save grouped by error type
            if 'failure_category' in df_failures.columns:
                for category in df_failures['failure_category'].unique():
                    category_df = df_failures[df_failures['failure_category'] == category]
                    sheet_name = f'Category_{category}'[:31]  # Excel sheet name limit
                    category_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Save grouped by model
            for model in df_failures['model'].unique():
                model_df = df_failures[df_failures['model'] == model]
                sheet_name = f'Model_{model}'[:31]
                model_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\nAnalysis complete! Results saved to:")
        print(f"  - Excel file: {output_path.absolute()}")
        print(f"  - Log file: {log_path.absolute()}")
        print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        # Restore standard output
        sys.stdout = original_stdout
        log_file_handle.close()
        print(f"\nLog file saved to: {log_path.absolute()}")

if __name__ == "__main__":
    main()


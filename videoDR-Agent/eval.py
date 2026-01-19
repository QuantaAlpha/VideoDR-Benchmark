#!/usr/bin/env python3
"""
Deep Researcher Single Sample Evaluation Script

Usage:
    python eval.py --index 4
    python eval.py --index 4 --profile gpt5_agentic
"""

import asyncio
import json
import argparse
import base64
import os
import sys
import logging
import re
import yaml
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from agentic_deep_research.deep_researcher import DeepResearcher

# Load .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from langchain_core.messages import HumanMessage

def load_config(profile_name=None):
    """Load and merge configuration"""
    if not os.path.exists("config.yaml"):
        raise FileNotFoundError(f"Configuration file config.yaml does not exist")
    
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    profile = profile_name or config.get("default_profile")
    if not profile or profile not in config.get("profiles", {}):
        raise ValueError(f"Profile '{profile}' not found in configuration")
    
    # Merge common configuration and specific profile configuration
    merged_config = config.get("common", {}).copy()
    merged_config.update(config["profiles"][profile])
    
    # Inject profile name for later use
    merged_config["profile_name"] = profile
    
    return merged_config

def setup_environment(config):
    """Setup environment variables"""
    os.environ["LANGCHAIN_TRACING_V2"] = config.get("langchain_tracing_v2", "false")
    
    env_vars = config.get("env", {})
    for key, value in env_vars.items():
        os.environ[key] = value

def pngs_to_base64(images_path: list[str]) -> str:
    """Convert multiple PNG image files to Base64 encoded strings (JSON format)"""
    if not isinstance(images_path, list):
        raise ValueError("Input must be a list of strings (image paths)")
    if len(images_path) == 0:
        raise ValueError("Image path list cannot be empty")
    
    png_base64_list = []
    for idx, img_path in enumerate(images_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {idx+1} does not exist: {img_path}")
        if not os.path.isfile(img_path):
            raise IOError(f"Path {idx+1} is not a file: {img_path}")
        
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        png_data_uri = f"data:image/png;base64,{img_base64}"
        png_base64_list.append(png_data_uri)
    
    return png_base64_list

def video_to_base64(video_path: str) -> str:
    """Convert video file to Base64 encoded string"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    return f"data:video/mp4;base64,{video_base64}"

def serialize_state(values: dict) -> dict:
    """Serialize LangGraph state to JSON"""
    if not values:
        return {}

    def _is_base64_uri(s: str) -> bool:
        return isinstance(s, str) and s.startswith("data:") and ";base64," in s

    def _truncate_base64(obj):
        if isinstance(obj, str):
            # Check if it's a data URI or just a raw base64 string that looks like an image
            if _is_base64_uri(obj):
                return obj[:30] + "...[TRUNCATED]"
            return obj
        elif isinstance(obj, list):
            return [_truncate_base64(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: _truncate_base64(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
             # Handle objects with attributes (like Pydantic models or arbitrary classes)
            return {k: _truncate_base64(v) for k, v in obj.__dict__.items()}
        else:
            return obj
    
    result = {}
    for key, value in values.items():
        # Apply truncation recursively to all values
        result[key] = _truncate_base64(value)
        
    return result

def sanitize_print_data(data):
    """Recursively process data, replacing Base64 encoded image strings with [base64_encode] for logging"""
    if isinstance(data, dict):
        return {k: sanitize_print_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_print_data(v) for v in data]
    elif isinstance(data, str):
        if data.startswith("data:image/") and ";base64," in data:
            return "[base64_encode]"
        elif data.startswith("data:video/") and ";base64," in data:
            return "[base64_encode]"
        return data
    elif hasattr(data, "content"):
        msg_dict = {
            "type": type(data).__name__,
            "content": sanitize_print_data(data.content)
        }
        if hasattr(data, "tool_calls") and data.tool_calls:
            msg_dict["tool_calls"] = sanitize_print_data(data.tool_calls)
        return msg_dict
    return data

class SanitizeLogFilter(logging.Filter):
    """Filter Base64 image data in logs"""
    def filter(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, str) and "Supervisor input messages:" in record.msg:
            record.msg = re.sub(r'data:image/[^;]+;base64,[a-zA-Z0-9+/=]+', '[base64_encode]', record.msg)
        return True

async def run_single(index: int, config: dict) -> dict:
    """Run single question"""
    mode = config["mode"]
    try:
        deep_researcher = DeepResearcher(mode=mode).graph
    except ValueError as e:
        raise ValueError(f"Mode must be 'workflow' or 'agentic', but got {mode}") from e

    csv_input = Path(config["csv_input"])
    if not csv_input.exists():
        print(f"[ERROR] CSV input file does not exist: {csv_input}")
        return

    df = pd.read_csv(csv_input, header=None, names=['index', 'question', 'answer'], encoding='utf-8', usecols=[0, 1, 2])
    row = df[df['index'] == index]
    if row.empty:
        print(f"[ERROR] index={index} does not exist")
        return
    
    row = row.iloc[0]
    question = row['question']
    video_format = config["video_format"]
    max_key_frame = config["max_key_frame"]
    
    # Build traj_dir: common.traj_dir / profile_name
    traj_base_dir = Path(config.get("traj_dir", "trajs"))
    profile_name = config.get("profile_name", "default")
    traj_dir = traj_base_dir / profile_name
    
    extra_config = config["extra_config"]

    if video_format == 'key_frame':
        video_dir = config['key_frames_dir']
        video_path = Path(video_dir)/f"{index}"
    else:
        video_dir = config['videos_dir']
        video_path = Path(video_dir)/f"{index}.mp4"
    
    print(f"\n{'='*50}")
    print(f"[{index}] {video_path}")
    print(f"Q: {question[:80]}..." if len(question) > 80 else f"Q: {question}")
    print(f"ground_truth: {row['answer'][:80]}..." if len(row['answer']) > 80 else f"ground_truth: {row['answer']}")
    print('='*50)
    
    traj_file = traj_dir / f"{index}.json"
    if os.path.exists(traj_file):
        print(f"[SKIP] Trajectory already exists: {traj_file}")
        return
    
    if video_format == 'mp4':
        if not str(video_path).startswith(('http', 'https')):
            if not os.path.exists(video_path):
                print(f"[ERROR] Video does not exist: {video_path}")
                return
            
            # Convert local file to Base64
            abs_path = video_path if os.path.isabs(video_path) else os.path.abspath(video_path)
            video_input = video_to_base64(abs_path)
        else:
            # Use remote URL directly
            video_input = video_path
            
        input_data = {
            "messages": [HumanMessage(content=question)],
            "video_url": video_input,
        }
    elif video_format == 'key_frame':
        if not os.path.exists(video_path):
             print(f"[ERROR] Key frame directory does not exist: {video_path}")
             return
        sorted_pngs = sorted(Path(video_path).glob("*.png"), key=lambda x: float(x.stem))
        assert len(sorted_pngs) > 0, f"[ERROR] Key frame directory is empty: {video_path}"
        # Remove random sampling, use uniform sampling or slicing to preserve order
        if len(sorted_pngs) > max_key_frame:
            # Simple uniform step sampling
            import numpy as np
            indices = np.linspace(0, len(sorted_pngs) - 1, max_key_frame, dtype=int)
            sorted_pngs = [sorted_pngs[i] for i in indices]
            
        sorted_pngs = pngs_to_base64([str(p) for p in sorted_pngs])
        input_data = {
            "messages": [HumanMessage(content=question)],
            "images_url": sorted_pngs,
        }
    
    run_config = {"configurable": {"thread_id": f"{mode}:{index}", **extra_config}}
    
    start_time = datetime.now()
    
    async for event in deep_researcher.astream(input_data, config=run_config, stream_mode="updates"):
        for node_name in event.keys():
            print(f"  -> {node_name}")
            print(sanitize_print_data(event[node_name]))
    
    final_state = await deep_researcher.aget_state(run_config)
    duration = (datetime.now() - start_time).total_seconds()
    
    # Extract answer
    answer = ""
    if final_state and final_state.values:
        messages = final_state.values.get("messages", [])
        if messages and hasattr(messages[-1], 'content'):
            answer = messages[-1].content
    
    print(f"[DONE] {duration:.1f}s")
    print(f"A: {answer[:100]}..." if len(answer) > 100 else f"A: {answer}")
    
    # Save trajectory
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_data = {
        "index": index,
        "video_url": video_path,
        "question": question,
        "answer": answer,
        "ground_truth": row['answer'],
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration,
        "values": serialize_state(final_state.values) if final_state and final_state.values else {},
        "metadata": final_state.metadata if final_state else {}
    }
    with open(traj_file, "w", encoding="utf-8") as f:
        json.dump(traj_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"[TRAJ] {traj_file}")
    
    return

def main():
    parser = argparse.ArgumentParser(description="Run Deep Researcher")
    parser.add_argument("--index", "-i", type=int, required=True, help="Index of the single question to run")
    parser.add_argument("--profile", "-p", type=str, help="Configuration profile to use (overrides default in config file)")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.profile)
    except Exception as e:
        print(f"[ERROR] Configuration load failed: {e}")
        return 1
    
    setup_environment(config)
    
    print(f"[RUN] Using Profile: {args.profile or 'default'} | Mode: {config['mode']}")
    print(f"[RUN] Running question index: {args.index}")
    
    # Add log filter
    logging.getLogger("agentic_deep_research.nodes.supervisor").addFilter(SanitizeLogFilter())
    
    asyncio.run(run_single(args.index, config))
    return 0

if __name__ == "__main__":
    exit(main())

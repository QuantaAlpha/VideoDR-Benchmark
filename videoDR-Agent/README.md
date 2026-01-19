# Agentic Deep Research with Video Capabilities 🎥🔍

This project implements an advanced deep research agent using LangGraph, capable of conducting in-depth research topics and analyzing video content.

## Features ✨

- **Agentic Workflow 🤖**: Uses LangGraph to manage state and coordination between Supervisor and Researcher agents.
- **Deep Research 📚**: Autonomous researcher agents that can use search tools (Tavily, etc.) to gather information.
- **Video Analysis 🎬**: Capable of processing video inputs (URLs or local files) to extract information relevant to the research topic.
- **Supervisor-Worker Architecture 🏗️**: A supervisor agent plans the research strategy and delegates tasks to multiple researcher workers.

## Installation 🛠️

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/simple-videoDR.git
   cd simple-videoDR
   ```

2. Install dependencies:
   
   This project uses `uv` for dependency management. You don't need to manually install dependencies if you use `uvx`.
   
   If you haven't installed `uv` yet:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   or

   ```bash
   pip install uv
   ```

## Configuration ⚙️

### Basic Configuration
The system core settings are in `src/agentic_deep_research/configuration.py`.

### Batch Running Configuration
For running batch evaluations or tests, use `config.yaml`. This file allows you to define different **profiles** for various scenarios (e.g., using different models, modes, or paths).

Example `config.yaml`:
```yaml
common:
  csv_input: "data/Video-DR.csv"
  traj_dir: "trajs"
  max_key_frame: 20
  langchain_tracing_v2: "false"
  videos_dir: "data/video"
  key_frames_dir: "data/video_key_frames"

default_profile: "qwen_workflow"

profiles:
  qwen_workflow:
    video_format: "mp4"
    mode: "workflow"
    extra_config:
      research_model: "openai:Qwen/Qwen3-Omni-30B-A3B-Instruct"
  
  gpt5_agentic:
    video_format: "key_frame"
    mode: "agentic"
    extra_config:
      research_model: "openai:openai/gpt-5.2"
```

## Data Preparation 📂

### Dataset
We use the **VideoDR** dataset. You can download it from [HuggingFace](https://huggingface.co/datasets/Yu2020/VideoDR).
- Place `VideoDR.csv` in the `data/` directory.
- Unzip `video.zip` to your video directory `data/videos`.

### CSV Format
The `eval.py` script reads the first 3 columns of the CSV file:
1. `index`: Unique identifier for the question/video.
2. `question`: The research question.
3. `answer`: The ground truth answer to the question.

### Video Format
- **mp4**: The system expects `{index}.mp4` in the `videos_dir` (default: `data/video`).
- **key_frame**: The system expects a directory named `{index}` in `key_frames_dir` (default: `data/video_key_frames`), containing `.png` files.
  - **Naming Convention**: Images must be named by their timestamp (e.g., `12.43.png` for 12.43 seconds).
  - **Order Matters**: The system sorts images by timestamp to ensure the model understands the temporal sequence. This format is useful for models that don't support direct video input.

## Usage 🚀

### Evaluation
Use `uvx` to run the evaluation script. This will automatically set up the environment with necessary dependencies.

```bash
# Evaluate a specific index using the default profile
uvx --with-editable . --python 3.11 python eval.py --index 4

# Evaluate with a specific profile defined in config.yaml
uvx --with-editable . --python 3.11 python eval.py --index 4 --profile gpt5_agentic
```

### Batch Processing (Parallel) ⚡
You can use `xargs` with `uvx` to run multiple evaluations in parallel.

```bash
# Run indices 1 to 100 with 10 concurrent processes
seq 1 100 | xargs -P 10 -I {} uvx --with-editable . --python 3.11 python eval.py --index {} --profile gpt5_agentic
```

The results (trajectories) will be saved in the `trajs/<profile_name>` directory.

### Library Usage
The core logic is in `src/agentic_deep_research/deep_researcher.py`. You can import `DeepResearcher` class to run the agent programmatically.

```python
from agentic_deep_research.deep_researcher import DeepResearcher
from agentic_deep_research.state import AgentInputState
from langchain_core.messages import HumanMessage

input_state = AgentInputState(
    messages=[HumanMessage(content="Research the impact of AI on video production")],
    video_url="path/to/video.mp4" # Optional
)

# Use agentic mode
agent = DeepResearcher(mode="agentic").graph
async for event in agent.astream(input_state):
    print(event)
    
# Use workflow mode
workflow = DeepResearcher(mode="workflow").graph
async for event in workflow.astream(input_state):
    print(event)
```

## Environment Variables 🔐
Set your API keys in your environment variables or `.env` file (see `env.example`):
- `OPENAI_API_KEY`: For OpenAI models.
- `TAVILY_API_KEY`: For web search capabilities.

**Note on LangSmith Tracing**: `LANGCHAIN_TRACING_V2` defaults to `false` in `env.example`. This is recommended because video inputs (Base64 encoded) can create very large trace payloads that may fail to upload or consume excessive quota.

## License 📜

[MIT](LICENSE)

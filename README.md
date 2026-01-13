# VideoDR: A Video Deep Research Benchmark on Open Web for Agentic Video Reasoning

<p align="center">
  <img src="https://img.shields.io/badge/Task-Video_Deep_Research-C92D39" alt="Task">
  <img src="https://img.shields.io/badge/Paradigm-Agentic_%26_Workflow-C92D39" alt="Paradigm">
  <img src="https://img.shields.io/badge/Benchmark-VideoDR-007EC6" alt="Benchmark">
</p>

![VideoDR Task Overview](./assets/6963e7b13de6cf860976558e.png)

<font size=7><div align='center' > [[📖 Paper](http://arxiv.org/abs/2601.06943)] [[📊 Dataset](https://huggingface.co/datasets/Yu2020/VideoDR)] </div></font>

---

# 🔥 News
* **`2026.01.12`** 🌟 We release VideoDR benchmark data. You can download it from [there](https://huggingface.co/datasets/Yu2020/VideoDR).
* **`2026.01.11`** 🌟 We are very proud to launch VideoDR, the first-ever video deep research benchmark!



# 🎥 Video Deep Research | VideoDR

🚀 **VideoDR** is the **first** video deep research benchmark!

It is designed to evaluate the capability of Multimodal Large Language Models to perform complex reasoning based on video content while leveraging the Open Web 🌐.

#### 👇 VideoDR requires the Agent to possess the following core capabilities:

* 🎞️ **Multi-frame Visual Cues**: Accurately identify continuous key information from multiple video frames.
* 🌍 **Interactive Search**: Interact with a browser environment to perform multi-hop deep search.
* 🧩 **Evidence Synthesis**: Combine video clues and web evidence to provide a **verifiable** factual answer.



# 📊 Eval Pipeline



# 🔧 Failure Analysis Tool

We provide an LLM-based failure analysis tool (`llm_as_judge`) to automatically classify failure cases into different error categories based on trace analysis.

### Installation

```bash
cd llm_as_judge
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the `llm_as_judge` directory with your LLM API credentials:

```bash
LLM_BASE_URL=your_api_base_url
LLM_API_KEY=your_api_key
```

### Usage

```bash
# Analyze all models
python llm_as_judge/src/analyze_failure_types.py \
    --excel_file llm_as_judge/data/Video-LLM.xlsx \
    --trace_dir results/traces \
    --max_workers 4

# Analyze specific models
python llm_as_judge/src/analyze_failure_types.py \
    --excel_file llm_as_judge/data/Video-LLM.xlsx \
    --trace_dir results/traces \
    --models gpt52 gpt4o \
    --max_workers 4
```

**Key Parameters:**
- `--excel_file`: Evaluation Excel file (default: `Video-LLM.xlsx`)
- `--trace_dir`: Directory with trace JSON files (default: `traces`)
- `--models`: Model keys to analyze: `qwen`, `internvl`, `minicpm`, `gpt4o`, `gemini`, `gpt52` (default: all)
- `--max_workers`: Concurrent workers (default: 4)

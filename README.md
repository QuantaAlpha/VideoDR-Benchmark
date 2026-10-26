# VideoDR: A Video Deep Research Benchmark on Open Web for Agentic Video Reasoning

<p align="center">
  <img src="https://img.shields.io/badge/Task-Video_Deep_Research-C92D39" alt="Task">
  <img src="https://img.shields.io/badge/Paradigm-Agentic_%26_Workflow-C92D39" alt="Paradigm">
  <img src="https://img.shields.io/badge/Benchmark-VideoDR-007EC6" alt="Benchmark">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

![VideoDR Task Overview](./assets/6963e7b13de6cf860976558e.png)

<font size=7><div align='center' > [[📖 Paper](http://arxiv.org/abs/2601.06943)] [[📊 Dataset - 500](https://huggingface.co/datasets/strike20023/VideoDR)] [[📁 Legacy Dataset - 100](https://huggingface.co/datasets/Yu2020/VideoDR)] [[🌐 LeaderBoard](https://videodr-benchmark.github.io/#/home)] </div></font>

---

# 🔥 News
* **`2026.05.19`** 🤖 We have expanded the benchmark set to 500 samples. You can find it in [there](https://huggingface.co/datasets/strike20023/VideoDR).
* **`2026.01.19`** 🤖 We have updated our code on the workflow and the agentic paradigm. You can find it in [there](https://github.com/QuantaAlpha/VideoDR-Benchmark/tree/main/videoDR-Agent).
* **`2026.01.15`** 🌐 Our [Official LeaderBoard](https://videodr-benchmark.github.io/#/home) is now live! Welcome to test and submit.
* **`2026.01.14`** 🏷️ We update `VideoDR.csv` with additional `Category` and `Difficulty` labels.
* **`2026.01.12`** 🌟 We release VideoDR benchmark data. You can download it from [there](https://huggingface.co/datasets/Yu2020/VideoDR).
* **`2026.01.11`** 🌟 We are very proud to launch [VideoDR](http://arxiv.org/abs/2601.06943), the first-ever video deep research benchmark!



# 🎥 Video Deep Research | VideoDR

🚀 **VideoDR** is the **first** video deep research benchmark!

It is designed to evaluate the capability of Video Agent to perform complex reasoning based on video content while leveraging the Open Web 🌐.

#### 👇 VideoDR requires the Agent to possess the following core capabilities:

* 🎞️ **Multi-frame Visual Cues**: Accurately identify continuous key information from multiple video frames.
* 🌍 **Interactive Search**: Interact with a browser environment to perform multi-hop deep search.
* 🧩 **Evidence Synthesis**: Combine video clues and web evidence to provide a **verifiable** factual answer.


# 📊 Current Dataset

The current benchmark is the **500-example** [Hugging Face dataset](https://huggingface.co/datasets/strike20023/VideoDR), containing `test.csv` and `videos/1.mp4` through `videos/500.mp4`. The 100-example release linked above is historical. See [data preparation](videoDR-Agent/README.md) for running the current dataset.

The historical 100-example manual evaluation workbook is no longer bundled. It contained model outputs and correctness annotations, not the complete current benchmark.

# 🗂️ Repository Structure

| Path | Contents |
| --- | --- |
| [`videoDR-Agent/`](videoDR-Agent/) | LangGraph-based agentic and workflow baselines, configuration, and evaluation runner |
| [`llm_as_judge/`](llm_as_judge/) | Answer judging and failure analysis tools |
| [`assets/`](assets/) | Figures and tables for the benchmark |

# 🚀 Quick Start

Use **Python 3.10 or later**. Run the following commands to install the baseline:

```bash
git clone https://github.com/QuantaAlpha/VideoDR-Benchmark.git
cd VideoDR-Benchmark/videoDR-Agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp env.example .env
```

1. Set `OPENAI_API_KEY` and `TAVILY_API_KEY` in `.env`. The bundled profiles use OpenRouter's OpenAI-compatible endpoint, so `OPENAI_API_KEY` must contain a key for the configured provider.
2. Download `test.csv` from the [current dataset](https://huggingface.co/datasets/strike20023/VideoDR) to `data/test.csv`, and place the downloaded videos in `data/video/` as `1.mp4` through `500.mp4`.
3. Review the research and summarization models and their endpoints in [`config.yaml`](videoDR-Agent/config.yaml) and [the configuration defaults](videoDR-Agent/src/agentic_deep_research/configuration.py).

Run one example from `videoDR-Agent/`:

```bash
# Agentic baseline
python eval.py --index 4 --profile gemini_agentic

# Workflow baseline
python eval.py --index 4 --profile gemini_workflow
```

Trajectories are saved to `trajs/<profile_name>/<index>.json`, including the question, generated answer, ground truth, and execution state. Existing trajectory files are skipped. See the [agent guide](videoDR-Agent/README.md) for key-frame inputs, custom profiles, and parallel evaluation.

# 🔧 Evaluation Tools

We provide LLM-based evaluation tools (`llm_as_judge`) for model evaluation and failure analysis.

Run the commands in this section from the repository root.

### Installation

```bash
python -m pip install -r llm_as_judge/requirements.txt
```

### Configuration

Create a `.env` file in the `llm_as_judge` directory:

```bash
LLM_BASE_URL=your_api_base_url
LLM_API_KEY=your_api_key
```

The judge currently uses the model name `/model/DeepSeek-V3` in [`judge_answers.py`](llm_as_judge/src/judge_answers.py). Use an endpoint that serves this model, or update the model name to match your endpoint.

### Prediction Format

The judge expects a JSON object keyed by question ID. Each question contains its reference answer and predictions grouped by model and mode:

```json
{
  "1": {
    "question": "Example question",
    "ground_truth": "Reference answer",
    "model_name": {
      "my-model": {
        "agentic": {"answer": "Model answer"},
        "workflow": {"answer": "Model answer"}
      }
    }
  }
}
```

Include only the modes you evaluated (`agentic` and/or `workflow`). Aggregate the runner's per-example trajectory files into this format before judging; an individual trajectory file is not a `predictions.json` file.

### LLM as Judge

```bash
python llm_as_judge/src/judge_answers.py \
    --workers 5 \
    --predictions path/to/predictions.json
```

The judge writes per-answer decisions and reasoning to `judge_results.csv`, with an Excel copy and a text report. **The summary report currently uses a fixed denominator of 100.** For the 500-example release, calculate accuracy from the per-answer results using the actual evaluation size, and report any missing or failed examples separately.

### Failure analysis

```bash
python llm_as_judge/src/analyze_failure_types.py \
    --excel_file path/to/manual_evaluation.xlsx \
    --trace_dir results/traces \
    --max_workers 4
```

Failure analysis requires a separately supplied manual evaluation workbook and matching traces; the benchmark CSV is not a replacement for this workbook. The workbook uses a grouping row, headers on the second row, then positional ID, URL, question, label, reference-answer, model-answer and correctness columns as defined in `load_manual_evaluation`.

# 📚 Citation

If you find this benchmark useful for your research, please cite:

```bibtex
@article{liu2026watching,
  title={Watching, Reasoning, and Searching: A Video Deep Research Benchmark on Open Web for Agentic Video Reasoning},
  author={Liu, Chengwen and Yu, Xiaomin and Chang, Zhuoyue and Huang, Zhe and Zhang, Shuo and Lian, Heng and Wang, Kunyi and Xu, Rui and Hu, Sen and Hou, Jianheng and others},
  journal={arXiv preprint arXiv:2601.06943},
  year={2026}
}
```

# 📄 License

The code in this repository is licensed under the [MIT License](LICENSE).

External datasets, source videos, and other third-party materials are subject to their respective licenses and terms. The code license does not grant rights to those materials; consult the dataset pages and original sources before reuse.

# ✉️ Contact


**Have a question?** If you have any questions or just want to say hi, feel free to reach out:

📧 **Email:** [yuxm02@gmail.com](mailto:yuxm02@gmail.com)

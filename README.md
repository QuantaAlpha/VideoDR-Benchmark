# VideoDR: A Video Deep Research Benchmark on Open Web for Agentic Video Reasoning

<p align="center">
  <img src="https://img.shields.io/badge/Task-Video_Deep_Research-C92D39" alt="Task">
  <img src="https://img.shields.io/badge/Paradigm-Agentic_%26_Workflow-C92D39" alt="Paradigm">
  <img src="https://img.shields.io/badge/Benchmark-VideoDR-007EC6" alt="Benchmark">
</p>

![VideoDR Task Overview](./assets/6963e7b13de6cf860976558e.png)

<font size=7><div align='center' > [[📖 Paper](http://arxiv.org/abs/2601.06943)] [[📊 Dataset](https://huggingface.co/datasets/Yu2020/VideoDR)] [[🌐 LeaderBoard](https://videodr-benchmark.github.io/#/home)] </div></font>

---

# 🔥 News

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


# 🔧 Evaluation Tools

We provide LLM-based evaluation tools (`llm_as_judge`) for model evaluation and failure analysis.

### Installation

```bash
cd llm_as_judge
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the `llm_as_judge` directory:

```bash
LLM_BASE_URL=your_api_base_url
LLM_API_KEY=your_api_key
```

### LLM as Judge

```
python llm_as_judge/src/judge_answers.py \
    --workers 5 \
    --predictions llm_as_judge/data/predictions.json
```

### Failure analysis

```
python llm_as_judge/src/analyze_failure_types.py \
    --excel_file llm_as_judge/data/Video-LLM.xlsx \
    --trace_dir results/traces \
    --max_workers 4
```

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

# ✉️ Contact


**Have a question?** If you have any questions or just want to say hi, feel free to reach out:

📧 **Email:** [yuxm02@gmail.com](mailto:yuxm02@gmail.com)

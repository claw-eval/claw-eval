<div align="center">

<img src="claw_eval.png" width="160" alt="Claw-Eval Logo">

# Claw-Eval

[![Tasks](https://img.shields.io/badge/tasks-300-blue)](#tasks)
[![Models](https://img.shields.io/badge/models-14-green)](#leaderboard)
[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2604.06132v1)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-purple)](https://claw-eval.github.io)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/claw-eval/Claw-Eval)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

> End-to-end transparent benchmark for AI agents acting in the real world. <br>
> 300 human-verified tasks | 2,159 rubrics | 9 categories | Completion · Safety · Robustness.

</div>


---

## Leaderboard

Browse the full leaderboard and individual task cases at **[claw-eval.github.io](https://claw-eval.github.io)**.

**Evaluation Logic (Updated March 2026):**

* **Primary Metric: Pass^3.** To eliminate "lucky runs," a model must now consistently pass a task across **three independent trials** ($N=3$) to earn a success credit.
* **Strict Pass Criterion:** Under the Pass^3 methodology, a task is only marked as passed if the model meets the success criteria in **all three runs**.
* **Reproducibility:** We are committed to end-to-end reproducibility. Our codebase is currently being audited to ensure **all benchmark results on the leaderboard can be verified by the community**.
* **Handling API Instability**: In the event of execution errors caused by network or API fluctuations, we manually re-trigger the evaluation to ensure exactly **3** trajectories are successfully generated.


## 📢 Updates
* **v1.1.0** — 35 multimodal agentic tasks + 38 multi-turn dialogue tasks. Agents perceive, reason, create, and deliver.

<img src="claweval_multimodal.png" width="1080" alt="Claw-Eval Multimodal">

* **v1.0.0** — Built on reproducible real-world complexity.
* **v0.0.0** — From chatbot to real world. (2026.3)



## Tasks

300 tasks across 3 splits and 9 categories, each task with human-verified rubrics.

| Split | Count | Description |
|-------|-------|-------------|
| `general` | 161 | Core agent tasks across communication, finance, ops, productivity, etc. |
| `multimodal` | 101 | Perception and creation — webpage generation, video QA, document extraction, etc. |
| `multi_turn` | 38 | Conversational tasks with simulated user personas for clarification and advice |

Agents are graded on three dimensions through full-trajectory auditing:
- **Completion** — did the agent finish the task?
- **Safety** — did it avoid harmful or unauthorized actions?
- **Robustness** — does it pass consistently across multiple trials?

### Dataset

Available on Hugging Face: [claw-eval/Claw-Eval](https://huggingface.co/datasets/claw-eval/Claw-Eval)

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Unique task identifier |
| `query` | string | Task instruction / description |
| `fixture` | list[string] | Fixture files required (available in `data/fixtures.tar.gz`) |
| `language` | string | `en` or `zh` |
| `category` | string | Task domain |

---

## Quick Start

We recommend using [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management:

```bash
pip install uv
uv venv --python 3.11
source .venv/bin/activate
```

Prepare your keys and set up the environments with one command:

```bash
export OPENROUTER_API_KEY=sk-or-...
export SERP_DEV_KEY=... # add this for tasks need real web search
bash scripts/test_sandbox.sh
```

> **Note on video fixtures:** Due to file size limits, this GitHub repository does not include video files for video-related tasks. The complete fixtures (including all videos) are available on Hugging Face: [claw-eval/Claw-Eval](https://huggingface.co/datasets/claw-eval/Claw-Eval).

Go rock 🚀

```bash
claw-eval batch --config model_configs/claude_opus_46.yaml --sandbox --trials 3 --parallel 16
```

---

## Roadmap

- [ ] More real-world, multimodal tasks in complex productivity environments
- [ ] Comprehensive, fine-grained scoring logic with deep state verification
- [ ] Enhanced sandbox isolation and full-trace tracking for transparent, scalable evaluation


## Contribution
We welcome any kind of contribution. Let us know if you have any suggestions!

## Acknowledgements
Our test cases are built on the work of the community. We draw from and adapt tasks contributed by OpenClaw, PinchBench, OfficeQA, OneMillion-Bench, Finance Agent, and Terminal-Bench 2.0.

## Core Contributors
[Bowen Ye](https://github.com/pkuYmiracle)(PKU), [Rang Li](https://github.com/lirang04) (PKU), [Qibin Yang](https://github.com/yangqibin-caibi) (PKU), [Zhihui Xie](https://zhxie.site/)(HKU), [Yuanxin Liu](https://llyx97.github.io/)(PKU), [Linli Yao](https://yaolinli.github.io/)(PKU), [Hanglong Lyu](https://github.com/Albus2002)(PKU), [Lei Li](lilei-nlp.github.io)(HKU, project lead)


## Advisors
[Tong Yang](https://yangtonghome.github.io/) (PKU), [Zhifang Sui](https://cs.pku.edu.cn/info/1226/2014.htm) (PKU), [Lingpeng Kong](https://ikekonglp.github.io/) (HKU), [Qi Liu](https://leuchine.github.io/) (HKU)

## Citation

If you use Claw-Eval in your research, please cite:

```bibtex
@misc{claw-eval2026,
  title={Claw-Eval: End-to-End Transparent Benchmark for AI Agents in the Real World},
  author={Ye, Bowen and Li, Rang and Yang, Qibin and Xie, Zhihui and Liu, Yuanxin and Yao, Linli and Lyu, Hanglong and Li, Lei},
  year={2026},
  url={https://github.com/claw-eval/claw-eval}
}
```

## License

This project is released under the [MIT License](LICENSE).


<div align="center">

<img src="claw_eval.png" width="160" alt="Claw-Eval Logo">

# Claw-Eval

[![Tasks](https://img.shields.io/badge/tasks-104-blue)](#tasks)
[![Models](https://img.shields.io/badge/models-22-green)](#leaderboard)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-purple)](https://claw-eval.github.io)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

> End-to-end transparent benchmark for AI agents acts in the real world. <br>
> 104 tasks, 15 services, Docker sandboxes, and robust grading.

</div>

---

## Leaderboard

Browse the full leaderboard and individual task cases at **[claw-eval.github.io](https://claw-eval.github.io)**.


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
Our test cases are built on the work of the community. We draw from and adapt tasks contributed by OpenClaw, PinBench, OfficeQA, OneMillion-Bench, Finance Agent, and Terminal-Bench 2.0.

## Contributors
[Bowen Ye*](https://github.com/pkuYmiracle)(PKU), [Rang Li*](https://github.com/lirang04) (PKU), [Qibin Yang](https://github.com/yangqibin-caibi) (PKU), [Zhihui Xie](https://zhxie.site/)(HKU), [Lei Li](lilei-nlp.github.io)$^\dagger$(HKU, Project Lead)




<div align="center">

# 🦞 Claw-Eval

[![Tasks](https://img.shields.io/badge/tasks-104-blue)](#tasks)
[![Models](https://img.shields.io/badge/models-22-green)](#leaderboard)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-purple)](https://claw-eval.github.io)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

> End-to-end benchmark for AI agents acting as personal assistants —<br>
> 104 tasks, 15 mock enterprise services, Docker sandboxes, and deterministic grading.

</div>

---

## Roadmap

- [ ] Skills
- [ ] Image and video support
- [ ] Memory consistency check
- [ ] More real-world, multimodal tasks in complex productivity environments
- [ ] Comprehensive, fine-grained scoring logic with deep state verification
- [ ] Enhanced sandbox isolation and full-trace tracking for transparent, scalable evaluation

---

## Leaderboard

Browse the full leaderboard and individual task cases at **[claw-eval.github.io](https://claw-eval.github.io)**.


## Quick Start

Prepare your keys and setup the environments with one command:

```bash
export OPENROUTER_API_KEY=sk-or-...
export SERP_DEV_KEY=...
bash scripts/test_sandbox.sh
```

Go rock 🚀

```bash
agent-eval batch --config configs/opus_config.yaml --sandbox --trials 3 --parallel 60
```

## Contribution
We welcome any kind of contribution. Let us know if you have any suggestions!

## Acknowledgements
Our test cases are built on the work of the community. We draw from and adapt tasks contributed by OpenClaw, PinBench, OfficeQA, OneMillion-Bench, Finance Agent, and Terminal-Bench 2.0.

## Contributors
Bowen Ye*, Rang Li*, Qibin Yang, Lei Li $^\dagger$

Peking University & University of Hong Kong



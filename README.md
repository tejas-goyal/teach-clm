# teachclm

Course research project (CMU 11-768) building on [Context Language Models](https://arxiv.org/abs/2609.37725) (CLMs): models that manage their own context by editing it as a file.

## Goal

CLM's RL rewards every context edit with the final outcome of the episode, so each training sample costs a full rollout. We train edits on a **teacher's per-edit reward** instead: a teacher that knows the gold answer and the steps to reach it scores an edit as soon as it is made, and the rollout stops there. The target is the same accuracy as CLM's recipe for less compute.

Details: [`docs/proposal_v2.md`](docs/proposal_v2.md) (what and why), [`docs/implementation_plan.md`](docs/implementation_plan.md) (how, milestones M0–M10).

## Layout

```
docs/       proposal, implementation plan, literature survey, mentor notes
infra/      Modal apps
  serve_vllm.py    Qwen3.5-9B behind an OpenAI-compatible vLLM server
  retriever.py     Wikipedia-2018 e5 + FAISS retriever (the MEM1 / Search-R1 setup)
  skyrl_smoke.py   SkyRL step-wise GRPO + LoRA smoke test
scripts/    repro_contextbench.sh: the paper's CLM agent on ContextBench via Harbor
teachclm/   the package (empty so far; agent loop, tasks, metrics, reward, RL go here)
tasks/      todo.md (current state), lessons.md
```

Not committed: `.env` (API key), `third_party/ContextBench` (checkout at `fb533c1`), `runs/` (outputs).

## Setup

```bash
git clone https://github.com/RulinShao/ContextBench.git third_party/ContextBench
git -C third_party/ContextBench checkout fb533c1
uv sync                                    # Python 3.12; CLM is pinned at commit 4831982
echo "TEACHCLM_API_KEY=<key>" > .env       # must match the Modal secret teachclm-api-key
uv run modal deploy infra/serve_vllm.py    # both apps scale to zero when idle
uv run modal deploy infra/retriever.py
```

Each file in `infra/` documents its own commands at the top. The ContextBench script needs a running Docker daemon (Colima works).

## Built so far (M0–M2)

- **Setup:** pinned dependencies; the Wikipedia index and corpus (21M passages) on a Modal volume.
- **Serving:** Qwen3.5-9B on Modal with working tool calls; the retriever returns correct passages at roughly 0.8–1 s per query.
- **RL trainer check:** SkyRL step-wise GRPO with LoRA trains Qwen3.5-0.8B end to end. Prompts that rewrite earlier context are not tested yet.
- **Paper's agent, unmodified:** runs through Harbor against our served 9B. On a ContextBench slice (two seeds each): key-value store 1.00 / 1.00, needle retention 1.00 / 1.00, sudoku 1.00 / 1.00, log triage 0.67 / 0.00.

Open point: the paper reports no Qwen3.5-9B number on ContextBench (its 9B numbers are on BrowseComp-Plus), so the slice above is a pipeline check, not yet a matched reproduction.

## Roadmap

| Milestone | What |
|---|---|
| M2 (decision) | Reproduction target: a BrowseComp-Plus subset, or keep ContextBench as a pipeline check |
| M3 | Our own agent loop, checked against the paper's agent on the same tasks |
| M4 | MEM1 multi-question QA task and grader; check that the context cap forces edits |
| M5 | Baselines: full context, truncate, summarize, MEM1-style, CLM zero-shot, hand-written skills |
| M6 | Prompt evolution (ICL) |
| M7 | Label edits, audit the oracle, check the teacher against it |
| M8 | RL plumbing on a 2B model; measure 9B step time and cost |
| M9 | RL runs: CLM recipe (baseline) vs teacher-only reward |
| M10 | Which edit behaviors emerge during RL |

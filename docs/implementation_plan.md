Implementation plan (draft for approval)

Status: plan only. Nothing is implemented yet.
Revised 2026-11-10 to match proposal_v2.md: reproduction first, ICL before RL, and teacher reward replacing the final outcome reward in training.
Inputs: proposal_v2.md, Weiwei's notes (both rounds), the CLM repo at commit 4831982 (Oct 13 2026), the MEM1 repo, and a survey of RL frameworks (Nov 8 2026).

What changed in this revision

| Topic | Change |
|---|---|
| Reproduction | New first milestone: run the paper's released agent, unmodified, on a ContextBench slice and compare with the paper's zero-shot number. Our own loop is then checked on the same tasks. |
| Training reward | Our method trains on the teacher's per-edit score only. No final outcome reward in training. |
| Rollouts | In our method a rollout is cut at the edit: sample G edits from the same context, score each, compare them against each other. |
| Arms | A = CLM recipe (baseline, full rollouts). T = teacher only. H = hybrid (recommended, not confirmed). O = oracle only (optional). |
| Main metric | Compute to reach A's accuracy, not steps. |
| Gates | Two new gates after the task is built: the cap forces edits at the N we train on, and the 9B's zero-shot valid-edit rate is usable. |
| Oracle | Now a heuristic. Uses supporting passages where available and gets a hand audit. |
| Cost | 9B step time is measured during RL plumbing, so the budget decision is made before the expensive runs. |

---

1. What the CLM repo gives us

The repo is small (about 10K lines). It has four parts:

| Part | What it is | Do we use it? |
|---|---|---|
| clm_harness/context_utils/context_string.py | Turns the message list into the editable file (`[[CTX_TURN i role=…]]` blocks) and parses an edited file back into a valid message list. The system prompt and task are pinned. Stdlib only. | Yes, as is. This is the heart of CLM. |
| clm_harness/context_env/ (env.py, edit_gate.py) | One turn on the environment side: write the mirror file, run the bash command, read the file back, apply the fit/shrink gate, and build the receipt line. It only needs an object with exec / upload_file / download_file. | Yes, as is. We give it our own small sandbox (see below). |
| clm_harness/utils/budget.py, tokens.py, finish_policy.py | Context-budget nudges, rollback-and-retry, the final-turn notice, token counting, the submit marker. | Yes, as is. |
| clm_harness/clm_agent/prompts.yaml | The CLM system prompt (the "Managing your context" section). | Yes, word for word. This is the zero-shot CLM baseline. |
| clm_harness/flops_metrics/ + agent_trajectory_format/ | FLOPs with prefix-cache reuse, computed from per-call context snapshots (the ATIF-CTX format). Constants already exist for "9b", which matches Qwen3.5-9B's hybrid layout (8 full-attention layers, hidden size 4096). | Yes. This is how we measure efficiency. |
| clm_icl/ (evolve.py, note.py, propose.py) | The skill-evolution loop: run, build a contrastive note, have a proposer rewrite SKILL.md, gate on the dev split. You plug in your own tasks by subclassing TaskSource. | Yes, as is, through a ~40-line TaskSource adapter. This is our ICL track. |
| clm_harness/clm_agent/harness.py (ClmAgent, 825 lines) | The agent loop, tied to Harbor (Docker sandboxes) and litellm. | Run unmodified for the reproduction (M2). After that it is reference only: we write a thin loop of about 200 lines from the same parts (reason below). |
| clm_contextbench/ | A Harbor adapter for ContextBench. | Yes, for the reproduction (M2) and the parity check (M3). |
| clm_rl/ | No trainer. Two patches against ProRL-Agent-Server + Slime (Megatron, 64 H200). What matters is the advantage formula and its settings. | We take the formula and the hyperparameters, not the code. This is arm A. |
| suffix_cache_reuse/ | Serving speedup (SGLang/vLLM). | No. We count FLOPs analytically. |

The CLM RL recipe (from clm_rl/README.md). This is arm A, our baseline:
- Stepwise GRPO. Every LM call is its own training sample, and all of them get the trajectory's group-normalized reward.
- Efficiency channel. Among the successful trajectories in a group, `a_eff = clip((mean_flops − flops)/mean_flops, −1, 1)`. It is added only on turns whose only action is a context edit, with weight `w_eff = 0.25`. Failed trajectories and groups with fewer than two successes get 0.
- Settings: lr 1e-6, KL 0.01, clip 0.2/0.28, DAPO dynamic sampling, a 0.3 penalty for no tool call, a 0.2 penalty for format failures.

Why write our own thin loop instead of running ClmAgent for everything:
1. Harbor runs every episode in a Docker sandbox. That is awkward on Modal and too slow for RL rollouts. CLM's environment code only needs exec/upload/download, so a local per-episode sandbox is enough.
2. RL needs exact token IDs from the rollout engine. ClmAgent hard-codes litellm calls.
3. The baselines (truncate, summarize, MEM1-style) need a different system prompt and a harness-side hook. ClmAgent reads both from module globals.
4. Our method has to stop an episode at an edit, sample several edits from that same point, and continue from one of them. ClmAgent runs an episode start to finish.

One loop with pluggable policies keeps eval, ICL and RL on exactly the same code. The parity check in M3 keeps us honest against upstream.

---

2. Codebase layout (minimal, extensible)

teachclm/
  pyproject.toml        # deps: clm @ git+…/context-language-models@4831982, litellm, datasets, modal
  teachclm/
    trajectory.py       # THE shared record (see below). Every other module reads/writes this.
    agent/
      loop.py           # step(state) -> state; run_episode(...) -> Trajectory; run_until_edit(...) -> state
      sandbox.py        # LocalSandbox: per-episode dir, async exec/upload_file/download_file, copy() for branching
      lm.py             # LMClient protocol; OpenAICompat (vLLM on Modal / API); later RLEngine
      policies.py       # context strategies, one class each:
                        #   Full, Truncate, Summarize, Mem1Loop  (harness-side)
                        #   Clm (mirror file; uses clm_harness ContextEnv)  + skill text
    tasks/
      base.py           # Task(id, instruction, gold, steps, meta) + grade(task, final_answer) -> reward
      contextbench.py   # the paper's tasks through our loop, for the parity check only
      mem1_qa.py        # build N-question tasks from FlashRAG HotpotQA/NQ; EM/F1 per sub-question
      search_cli.py     # `search "query"` command available inside the sandbox -> retriever HTTP
    metrics/
      flops.py          # thin wrapper over clm_harness.flops_metrics
      edits.py          # extract each edit (before/after diff); oracle label; behavior tag
    reward/
      teacher.py        # LLM teacher: (question, gold answers, steps, edit diff) -> {+1, 0, -1}
    icl/
      source.py         # clm_icl.TaskSource adapter -> reuse clm_icl.Evolution unchanged
    rl/                 # built only after the eval tracks are done
      edit_groups.py    # arms T/H/O: run to an edit point, sample G edits, score, continue from one
      stepwise.py       # arm A: SkyRL step-wise generator that calls run_episode
      advantage.py      # pure functions, unit tested: clm_recipe (A), group-normalized edit score (T/O), hybrid (H)
  infra/                # Modal apps
    serve_vllm.py       # Qwen3.5-9B OpenAI-compatible server (tool + reasoning parsers)
    retriever.py        # e5 + FAISS wiki-18 index on a Volume, HTTP /retrieve
    run_eval.py         # fan out episodes over a task list x policy grid -> Volume
  configs/              # one YAML per experiment (model, budget, N, policy, skill, arm, seeds)
  analysis/             # notebooks / plot scripts (acc-vs-FLOPs, acc-vs-compute, behavior frequencies)
  tests/                # parity with ClmAgent, context_string round-trip, grader, advantage, branching

The one design rule: everything is joined by a single Trajectory record. Each LM call stores:
- the exact prompt messages
- the response (plus token IDs and logprobs, when the client provides them)
- whether the turn was edit-only
- the edit diff and the context size before and after

The episode stores the final answer, the reward and the FLOPs.

Because of that, each kind of extension lives in one place:
- a new context strategy is one class in policies.py
- a new task is one file in tasks/
- a new reward signal is one function that reads a Trajectory or an edit diff
- both RL paths call the same loop

Branching. Our method needs to stop at an edit, try G edits from the same point, and carry on from one. So the loop is written as a step function over an explicit state (messages plus sandbox), and LocalSandbox can be copied. run_episode is just step in a loop. This is the one structural addition over the Nov 8 layout.

LocalSandbox detail: the CLM prompt hard-codes `/tmp/.live_ctx/LIVE_CTX_MAIN.txt`. LocalSandbox maps that path, and `/tmp/.bash_ctx_state`, to a private directory per episode. The model sees the paper's exact prompt, while many episodes run side by side in one Modal container. Modal containers are already isolated, so we don't need Docker.

---

3. The task: MEM1 multi-objective QA

- **Data:** HF RUC-NLPIR/FlashRAG_datasets, NQ and HotpotQA. N questions from the same source are joined into one task, as in MEM1's qa_search_*_merge_multi.py. We regenerate the tasks ourselves so we keep the sub-questions separately (MEM1 only stores the joined string).
- **Grading:** EM and F1 per sub-question, as in MEM1's qa_multiple.py, without its None-check bug. We report both the EM count (MEM1's 0..N number) and the normalized fraction.
- **Retriever:** e5-base-v2 over Wikipedia-2018 (21M passages), top-3. This is the MEM1 / Search-R1 setup.
  - The index is a 65 GB download plus a 5 GB corpus, kept on a Modal Volume.
  - The plan is to serve it from a high-memory CPU container, about 70 GB RAM, which is cheap.
  - If latency is too high, we move it to one GPU (fp16, about 32 GB).
  - BM25 is the fallback.
- **Setting:**
  - Eval uses N ∈ {2, 4, 8, 16}.
  - The context budget must force editing at the N we train on, not only at N=8–16. A search returns roughly 450 tokens (top-3 passages of about 100 words; an estimate), so an N=2 episode under an 8K cap may never need an edit. M4 picks the training N and cap together; training at N=4–8 or a smaller cap are the two levers.
  - MEM1's paper numbers come from a different model (Qwen2.5-7B), so they are reference points, not targets.
- **Steps the teacher sees:** each sub-question with its gold answer, and for HotpotQA the supporting facts, if the FlashRAG copy keeps them (not yet checked; otherwise we join them from the original HotpotQA release).
- **Oracle (heuristic ground truth for each edit):** an edit is harmful if a gold answer string, or a supporting passage, was in the context before the edit and is gone after it, for a sub-question the agent had not yet answered or noted. Answer strings alone miss bridge passages and match short answers by accident, so about 100 oracle labels are hand-audited in M7.

---

4. Order of work (each milestone has an exit check)

Two tracks run in parallel after M3: Track A is eval and ICL (M4–M7), Track B is RL plumbing (M8).

M0. Setup
Repo skeleton, pyproject pinned to CLM commit 4831982, Modal secrets. Start the 70 GB wiki-index download to a Modal Volume first, because it is the slowest step. One-day smoke test of SkyRL step-wise mode with LoRA on Qwen3.5, because a failure there changes the trainer for arm A.
Exit: Volume populated; `pip install -e .` imports clm_harness leaf modules; SkyRL smoke test passes or the fallback (prime-rl) is chosen.

M1. Serving
vLLM Qwen3.5-9B on one H100 (`--tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only`) and the retriever service.
Exit: a tool call round-trips through the server, and `/retrieve` answers in under 1 s.

M2. Reproduce the paper's zero-shot result
Run upstream ClmAgent, unmodified, through Harbor and the clm_contextbench adapter on a ContextBench slice, pointed at our served 9B. Harbor needs Docker, so this runs on a machine with Docker that calls the Modal endpoint.
Exit: zero-shot CLM accuracy and FLOPs on the slice, set beside the paper's reported number, with the gap explained if there is one. If the repo also ships the Summarize or full-context baselines, run those too; otherwise only CLM zero-shot is reproduced here.

M3. Our agent loop + Clm policy + parity
LocalSandbox, loop.py (step function, branching), the Clm policy, the Trajectory record, and a ContextBench loader for our loop.
Exit:
- the first request is byte-identical to upstream ClmAgent (same system prompt hash)
- context_string round-trip tests pass
- on the M2 slice, our loop gives accuracy, edit counts and FLOPs close to the upstream run
- branching test: G continuations from a copied state do not interfere with each other

M4. MEM1-QA task + grader + two gates
Task builder, grader, search CLI.
Exit:
- the grader matches MEM1's scorer on its own examples
- Full-context Qwen3.5-9B at N=2 gives a sensible EM
- Gate 1: a training N and cap are chosen so that episodes at that N need edits, and Full clearly degrades at N=8–16
- Gate 2: the 9B's zero-shot valid-edit rate and edits per episode are measured. If edits are rare, RL will start from the best skill prompt from M5/M6.

M5. ICL part 1: baselines and hand-written skills
Policies Full / Truncate / Summarize / Mem1Loop / Clm zero-shot, plus Clm with hand-written skills (extract-then-delete, checkpoint-and-prune), crossed with N ∈ {2, 4, 8, 16}, about 200 tasks per cell.
Exit:
- an accuracy-vs-FLOPs plot per N
- does the paper's ordering show up on this task? The paper has CLM ≥ Summarize at fewer FLOPs for strong models, and the 9B trailing Summarize before RL. Either outcome is reported.

M6. ICL part 2: prompt evolution
Run clm_icl.Evolution through our TaskSource on train/dev splits, with a strong API model as proposer, then evaluate once on test.
Exit: the evolved skill compared with zero-shot CLM and the best hand-written skill. The best prompt overall is fixed as the starting prompt for every RL arm.

M7. Edit analysis + oracle audit + teacher validity
edits.py: extract every edit from the M5/M6 trajectories, attach the oracle label and a behavior tag (drops an observation / summarizes / keeps a notes ledger / prunes after a checkpoint / rewrites the plan). Hand-audit about 100 oracle labels. Then teacher.py: score the same edits, with the patterns that worked in M5/M6 written into the rubric.
Exit: oracle accuracy on the audit; teacher precision and recall on harmful edits against the oracle and the audit; frequency of each behavior per policy. This gates M9: the teacher is the only training signal in arm T, so a weak teacher is fixed here.

M8. RL plumbing on a small model (Track B, starts after M3)
Two paths, both on Qwen3.5-2B first:
- Arm A path: SkyRL step-wise mode with a generator around run_episode. The LM client wraps SkyRL's engine to get exact token IDs.
- Arm T path: edit_groups.py. Roll to an edit point, sample G edits from the same context, score them, group-normalize, continue from one. Each group is a single-turn GRPO sample, so this path does not depend on step-wise support.
Exit:
- each path overfits about 16 easy tasks (the 2B may fail a real task for model reasons, so "reward rises on the train split" is not the check)
- the token IDs of each sample match the prompt and response we logged
- the edit-turn mask is right on hand-checked samples
- a few steps of each path on the 9B, to measure step time, teacher-call latency and cost. The budget decision (seeds, arms, 9B or 4B) is made here.

M9. RL experiments
Qwen3.5-9B LoRA, same tasks and starting prompt for every arm. Order:
- Arm A, one seed, as the pilot. CLM recipe: full rollouts, outcome + success-gated a_eff.
- Arm T: teacher-only reward on edit groups. No outcome reward.
- Arm H (recommended; not yet confirmed): T plus a small outcome reward on a fraction of full rollouts.
- Arm O (optional): T with the oracle in place of the teacher.
- Ablation on T if budget allows: short playout to the next answered sub-question.
Every checkpoint of every arm is evaluated on full episodes with final-answer grading.
Exit: accuracy against compute (GPU-hours, tokens generated, teacher calls) for each arm, and compute to reach A's final accuracy.

M10. Emergent behaviors
Run the M7 tagger on rollouts from each RL checkpoint. Which edit behaviors appear or grow, and do they differ between arms?

Why this order:
- Reproduction comes first, so a strange result later can be traced to the new task and not to our setup.
- Every step depends only on earlier ones.
- The cheap results that are sure to come (M2, M5–M7) arrive before the expensive, risky one (M9).
- RL plumbing is debugged on a 2B model, where a failure costs dollars, not hundreds.
- The teacher is validated before it becomes the only training signal.

---

5. Cost (estimates, to be replaced by M1/M8 measurements)

| Item | Estimate |
|---|---|
| Reproduction (M2) | A ContextBench slice with one policy. Small next to M5; not yet estimated. |
| Eval and ICL (M4–M7) | 5–7 policies × 4 N × ~200 tasks, plus ICL rounds: roughly 30–60 H100-hours of vLLM serving, plus a CPU retriever. About $150–300. |
| Teacher calls (M7) | Diff plus answer only, a few hundred tokens each. Tens of dollars on a cheap API model. |
| RL plumbing, 2B (M8) | Tens of H100-hours. |
| Arm A, 9B LoRA (M9) | Extrapolated from Search-R1-style runs: about 200–600 H100-hours per run at 8–16K context and about 200 steps, i.e. $0.8k–2.4k per run (H100 ≈ $3.95/h). This is now the largest single cost. |
| Arms T / H / O (M9) | Not estimated. Rollouts are cut at the edit, so a step should cost far less than in A, but the teacher is called for every sampled edit inside the training loop. Measured in M8. |

If arm A on the 9B doesn't fit, the order of fallbacks is:
1. Fewer steps or a shorter context.
2. Fewer seeds for A.
3. Qwen3.5-4B for RL only, keeping 9B for eval and ICL. FLOPs constants for 4B would need to be added.
If arms must be cut, drop O first, then the playout ablation. A and T are the comparison the project is about.

---

6. Open checks (could change the plan)
- Whether upstream ClmAgent runs end to end for us through Harbor against a Modal-served model, and which baselines the repo ships.
- How ContextBench tasks are packaged, and how much work the loader for our own loop is.
- Whether SkyRL step-wise + LoRA + Qwen3.5 works end to end. It isn't verified. prime-rl is the fallback; it needs a port to its `verifiers` env.
- Whether the FlashRAG HotpotQA copy keeps supporting facts.
- How often the 9B edits zero-shot (Gate 2), which sets how many edit points arm T has to train on.
- Teacher-call latency and cost inside the training loop.
- The vLLM version needed for Qwen3.5 tool parsing.
- CPU retriever latency on Modal.
- MEM1's test mix. Its code samples from 7 datasets, but the paper says HotpotQA+NQ. We will use HotpotQA+NQ and state that.
- Licenses: CLM is CC BY-NC 4.0 and MEM1 is MIT, which is fine for coursework. FlashRAG is CC-BY-SA.
- Decisions still needed from the user: approve this plan, total Modal budget, course deadline, and whether to include arm H.

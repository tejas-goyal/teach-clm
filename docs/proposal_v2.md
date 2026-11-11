Proposal v2: Training context edits from a teacher that knows the answer, without waiting for the final outcome

Revised 2026-11-10. This revision folds in Weiwei's second round of notes (MEM1 QA, 9B model, ICL first) and one design decision: in our method the teacher's reward replaces the final outcome reward during training. The previous text said the outcome reward stays the main term; that is no longer the plan.

The idea in three sentences

CLM's RL gives every context edit in a trajectory the same credit: the final outcome. So each training sample costs a full episode, and the signal for any single edit is noisy. A teacher that knows the gold answer and the steps to reach it can judge an edit as soon as it is made, so we train edits on the teacher's reward alone, stop the rollout at the edit, and aim to reach the accuracy of CLM's recipe with much less compute.

Why it should work (first principles)

Credit assignment. With outcome-only GRPO, the gradient on an edit is the gradient on the whole trajectory. Everything else that happened (search luck, reasoning errors) is noise on top of the edit's real effect.
Privileged information makes judging easy. Deciding "is this edit good?" is hard for the agent, which doesn't know the answer yet. It is easy for a judge that does: "the deleted text contained the passage the answer depends on" is close to a lookup.
Cheap samples. The teacher scores the edit itself. There is no counterfactual rerun, and no need to play the episode to the end to get a reward. A training sample costs one edit, not one episode.
We are not claiming a higher final score. Weiwei's point stands: outcome rewards do converge eventually. The claim is about cost: the same accuracy as CLM's recipe, for less compute, with fewer harmful deletions along the way.

What changed in this revision, and why

| Topic | Before (Nov 3) | Now (Nov 10) | Why |
|---|---|---|---|
| Training reward | Outcome reward as the main term, plus a per-edit teacher term | Teacher reward only. No final outcome reward in training. | Design decision: if the teacher knows the answer and the steps, it can reward an edit early. |
| Rollouts in training | Full episodes | Cut at the edit (optionally a short playout to the next intermediate step) | Follows from the reward change. This is where the saving comes from. |
| Main metric | RL steps to reach the baseline's accuracy | Compute to reach the baseline's accuracy | Our steps are much cheaper than the baseline's, so steps are not comparable. |
| Model | 1–2B, with 9B as a reference | Qwen3.5-9B. Qwen3.5-2B only to debug RL plumbing. | Weiwei: 1–2B may not pass at all; use 8–9B as in the paper. |
| Task | ContextBench synthetic tasks | MEM1's multi-question QA with a search tool | Weiwei: start from MEM1's QA setup; be careful with ContextBench because it is synthetic. |
| Oracle | Exact, from synthetic task labels | Heuristic, from gold answers and supporting passages; hand-audited | Real retrieval has no exact labels. |
| Order | RL infra and strategy comparison in parallel from week 1 | Reproduce the paper's zero-shot result, then ICL, then RL | Weiwei: ICL is a good starting point. Reproduction tells us our setup is right before we change anything. |
| Emergent behaviors | Not included | Tag each edit by type and track the types across RL checkpoints | Weiwei: RL can make good editing behaviors emerge; look for them. |

Setup

Model: Qwen3.5-9B for all results. Qwen3.5-2B only for debugging the RL code.
Tasks: MEM1's multi-question QA. N questions from HotpotQA or NQ are joined into one task, answered with a search tool, and graded by exact match per sub-question. The context cap is set so the model has to edit to succeed. ContextBench is used only for the reproduction check in Part 0.
Env: CLM's released code (facebookresearch/context-language-models) for the context-as-file environment. No Suffix Cache Reuse; FLOPs are counted analytically.
Trainer: SkyRL for the baseline arm (CLM's stepwise GRPO). The teacher arm trains on single edit turns, so it needs only ordinary single-turn GRPO.

Part 0: Reproduce the paper's zero-shot result (no training)

Run the paper's released agent code, unmodified, with Qwen3.5-9B on a slice of ContextBench, and compare against the paper's reported zero-shot CLM number. Then run our own agent loop on the same tasks and check that it matches. This is the only place we compare against a number from the paper. We do not try to reproduce the paper's RL numbers: the released RL setup targets a 64-H200 cluster.

Part 1: ICL. Which edit strategies work, with prompting only?

Run the same model under fixed strategies on MEM1 QA and compare accuracy against FLOPs:

Full context (no management)
Truncate at the cap
Summarize near the cap (CLM's main baseline)
MEM1-style loop
Free-form CLM editing (the paper's default prompt)
Extract-then-delete: copy needed facts into a notes block, then delete raw observations
Checkpoint-and-prune (Weiwei's pattern): once an intermediate result is confirmed, keep the result and delete the steps that produced it

Then run CLM's prompt-evolution loop on our tasks to get an evolved skill prompt.

Output: an accuracy-vs-FLOPs plot per strategy and N, plus the harmful-deletion rate. This part feeds RL in two ways: the edit patterns that work go into the teacher's rubric, and the best prompt becomes the starting prompt for RL. It is a result on its own even if RL fails.

Part 2: Is the teacher's per-edit signal right? (no training)

This part matters more than before, because the teacher is now the only training signal.

Two scorers, each giving an edit +1 / 0 / −1:

Oracle (heuristic). Before the edit, was the material a still-unanswered sub-question depends on in context? After the edit, is it still there anywhere (raw or in notes)? Lost it: −1. Removed a meaningful amount of text and lost nothing: +1. Otherwise 0. "Material" is the gold answer string and, for HotpotQA, the supporting passages, if the dataset copy we use keeps those labels (not yet checked). Answer strings alone miss bridge passages and match short answers like "yes" by accident, so we hand-audit about 100 oracle labels before trusting it.
Teacher. A strong LLM sees the question, the gold answer, the steps to get there (sub-question answers and supporting facts where available), and the edit diff. It uses a short rubric: Did the edit drop anything the answer still needs? Did it add anything false or any instruction the user never gave? Did it prune finished work?

We score a few thousand edits collected from Part 1 rollouts and report how well the teacher agrees with the oracle and with the hand audit (precision and recall on harmful edits). If agreement is poor, we fix the rubric here, before spending any RL compute.

Part 3: Does teacher-only reward reach the same accuracy for less compute? (main experiment)

How a training sample is built in our method:

def teacher_group(task, policy, teacher, G):
    state = run_until_edit(task, policy)            # context at the point where the model edits
    edits = [policy.sample(state) for _ in range(G)] # G different edits from the same context
    scores = [teacher.score(task.question, task.gold, task.steps, diff(state, e)) for e in edits]
    adv = group_normalize(scores)                    # compare the edits against each other
    return state, edits, adv                         # no final answer, no outcome reward
    # the episode then continues from one sampled edit to the next edit point

A sample that makes no edit scores 0.

Arms, same model, same tasks, same starting prompt:

A. Baseline: CLM's recipe. Full rollouts, final outcome reward plus the success-gated efficiency term. This is our own run of the paper's method.
T. Teacher only (our method). Rollouts cut at the edit. Reward is the teacher's score.
H. Hybrid (recommended; not yet confirmed). T, plus a small outcome reward on a fraction of full rollouts. If T drifts away from answering correctly, H shows whether a little outcome signal fixes it.
O. Oracle only (optional diagnostic). T with the oracle in place of the teacher. It separates "per-edit reward works" from "the teacher is good enough".

One ablation on T: a short playout. After the edit, continue until the next sub-question is answered and add whether that answer is correct. This is Weiwei's "play out more steps until an intermediate step is reached", and MEM1 QA gives a natural cut point because every sub-question has its own gold answer.

Metrics

Main: compute (GPU-hours and tokens generated, plus teacher calls) to reach arm A's final accuracy.
Final accuracy and FLOPs per task, measured on full episodes with final-answer grading for every arm. Outcome is used for measuring, not for training.
Harmful-deletion rate over training.
Frequency of each edit behavior across checkpoints.
Seeds: as many as the budget allows, decided after step time is measured. The claim is about speed, so one seed is not enough for the headline comparison.

Budget

Not yet measured. The estimates below come from the implementation plan and will be replaced by measurements.
Parts 0–2 (inference only): roughly $150–300.
Arm A on the 9B: a rough $0.8k–2.4k per run. This is now the expensive part of the project.
Arms T, H, O: expected to be much cheaper per step than A because rollouts are cut short, but the size of the saving is the thing we are measuring. Teacher calls now happen inside the training loop, so their cost and latency need measuring too.
If A does not fit: fewer steps, then fewer seeds, then run RL on Qwen3.5-4B.

Order of work

1. Part 0: reproduce the paper's zero-shot result.
2. Build MEM1 QA; Part 1: baselines, hand-written skills, prompt evolution.
3. Part 2: edit labels, oracle audit, teacher check.
4. RL plumbing on the 2B, in parallel with steps 2–3.
5. Part 3: arm A pilot, then T, then H, then O.
6. Edit behaviors across checkpoints; write-up.
Dates are to be set against the course deadline.

What we get if things go wrong

RL never stabilizes: Parts 0–2 are still a full report: a reproduction, which context strategies work on a real multi-question task, and whether a teacher with the answer can judge edits.
T does not reach A's accuracy: H and O say why. If H recovers it, the outcome reward is needed but only a little. If O works and T doesn't, the teacher is the problem and Part 2 shows where.
T reaches A's accuracy but not faster: a clean negative result on the compute claim.

Risks

Nothing in training ties the model to answering correctly. With no outcome reward, the model can learn edits the teacher likes that don't help the task. Mitigations: every checkpoint is evaluated on full episodes; arm H; arm O cannot be talked into a high score.
Only edit turns are trained. Searching, reasoning and answering get no signal in arm T, so they must already be good enough. Part 1 tells us whether they are.
The model rarely edits zero-shot. Then there are few edit points to train on. Gate after the task is built: measure the zero-shot valid-edit rate. If it is low, start RL from the best skill prompt from Part 1.
The training setting doesn't force edits. Small N with a generous cap never needs an edit. The cap is calibrated at the N we train on.
The model farms +1 with lots of tiny edits. +1 only counts for edits that remove a meaningful amount of text.
The model learns to never delete. A no-edit sample scores 0, and the context cap forces edits.
The oracle is wrong. It is a heuristic on real retrieval. Hand audit in Part 2.

Open checks (assumptions we have not verified yet)

Whether the paper's agent code runs for us end to end (it uses Docker sandboxes) and which baselines the repo ships.
Whether the HotpotQA copy we use keeps supporting-fact labels.
Whether SkyRL's step-wise mode works with LoRA on Qwen3.5 (needed for arm A).
Real step time for A and T on one H100, and teacher-call latency inside the training loop.
The course deadline and the total Modal budget.

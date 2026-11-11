The strongest proposal for you builds on a gap the CLM paper admits but doesn’t fix: no one teaches the model which context edits were good or bad. That puts it squarely in reward design, which is your area.

Lead proposal: edit-level credit assignment for Context Language Models

The gap. CLM trains with stepwise GRPO and gives every segment of a trajectory the same final outcome advantage. The authors say themselves that outcome rewards give only weak supervision for editing:

A successful run can contain bad edits that got lucky.
A failed run can contain good edits that got punished anyway.

They tried rewarding edit frequency or deleted volume and dropped it because the model games it. Their fix, the FLOPs-based efficiency bonus among successful runs, measures cost, not edit quality. Context-Folding went the other way with dense, hand-built penalties, but those only fit its rigid branch/return structure. Nobody has a dense reward for free-form edits. That’s the hole.

Hypothesis. An edit-level reward makes CLMs learn faster and make fewer harmful deletions, with the biggest gain on small models. In the paper, the 9B model lagged a plain summary harness by 6 points before RL.

Method: three signals, cheapest first.

Deletion regret (free, from hindsight). If the agent deletes something and later has to re-search, re-open, or re-derive it, that deletion was a mistake. You can detect this automatically by matching text in later observations against the deleted spans. You can also check whether the final answer relied on facts that were deleted at some point.
Hindsight rubrics (your rubric work). Once a trajectory finishes, you know what the answer actually needed. Use that to generate a per-task rubric for judging each edit, for example:
Does it keep verified facts?
Does it keep open TODOs?
Does it introduce instructions that didn’t come from the user?
Rubrics built from the end of the trajectory are new. CLM’s RL used a fixed, generic rubric only to grade the final answer.
Counterfactual replay (expensive, ablation only). Re-run from the context just before an edit, without that edit, and measure the change in success. This is your gold standard for checking whether signals 1 and 2 actually track edit quality.

The reward attaches only to edit-segment tokens. CLM already gives its efficiency term only to context-management tokens, so this slots in cleanly:

python
def edit_advantage(traj, group, w_eff=0.25, w_edit=0.5):
    A_out = grpo_outcome_adv(traj, group)           # CLM's existing term
    A_eff = success_gated_flops_adv(traj, group)    # CLM's existing term
    per_segment = []
    for seg in traj.segments:
        A = A_out
        if seg.is_context_edit:
            regret = deletion_regret(seg.deleted_spans, traj.later_observations)  # signal 1
            rubric = hindsight_rubric_score(seg, traj.final_answer)              # signal 2
            A += w_eff * A_eff + w_edit * (rubric - regret)
        per_segment.append(A)
    return per_segment

One design decision is worth arguing in the paper: whether the edit term should be group-normalized the way GRPO normalizes outcomes. You could compare an edit against edits at a similar step in sibling rollouts, rather than using raw scores. That choice is a real knob, not a default.

Why ContextBench makes this cheap to validate. Its four synthetic tasks have deterministic graders, and the right answer is fully determined. So you get ground-truth labels for whether each edit was harmful: a deleted needle or a lost KV value is objectively a mistake. You can measure how well your regret and rubric signals agree with those labels before spending any RL compute. That gives a clean section on whether the reward is valid, which reviewers like.

Experiments

Testbeds:
ContextBench for validating the signals and for cheap RL.
BrowseComp-Plus as the main benchmark, comparable to both papers.
TBLite for transfer to coding.
Baselines:
CLM’s own recipe (outcome + efficiency).
FoldGRPO-style penalties carried over to CLM.
A summary harness trained with the same recipe.
Metrics:
Accuracy and prefix-reuse FLOPs (CLM’s two metrics).
Plus new ones: edit precision (fraction of deletions never needed again) and re-retrieval rate.
Scale: CLM’s RL run used 64 H200s, so don’t try to match it. Run Qwen3.5-4B/9B on ContextBench and a slice of BrowseComp-Plus on Modal. The story that small models gain most from dense edit rewards fits a small compute budget anyway.

Risks and how to handle them

The model learns to never delete. The efficiency term already pushes against that; report the tradeoff curve.
Judge cost. Score only edit segments, which are a small fraction of tokens, and use a nano-sized judge, the way CLM did.
Getting scooped. CLM is from September 2026 and the code is open (facebookresearch/context-language-models), so other groups will move quickly. Aim for the ICML 2027 cycle.
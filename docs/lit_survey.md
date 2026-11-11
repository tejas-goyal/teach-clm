Paper 1: Context Language Models (CLMs), arXiv 2609.37725 (UW / Meta, Sep 2026)

Novelty, in one line: Give the model its own context as a file it can freely rewrite, instead of handing it a few fixed context-management tools.

Method, simply:

Context as a file. The model’s live context is mirrored to an editable file whose path is in the system prompt. The model edits it with ordinary Bash commands, and every edit is synced back as the model’s actual context for the next turn. If it doesn’t edit, new tokens are just appended as usual. So the model can delete old search results, rewrite a summary, keep a scoreboard, add a “notes” role, or write a reusable compact_turns() helper. 
arxiv
Works zero-shot. It needs no training to start. It beats summarization, MEM1, ACM, RLM and others on BrowseComp-Plus, TerminalBench, and 12–24h repo-optimization tasks, often with fewer FLOPs.
Can be taught in two ways.
Steering in plain English, e.g. “compact at subtask boundaries,” or evolving a “skill doc” with a prompt-evolution loop.
RL using stepwise GRPO plus a success-gated efficiency advantage. Among only the successful trajectories in a group, cheaper ones (by FLOPs) get a bonus. Failed ones get nothing, which keeps the model from gaming the reward with pointless edits. 
arxiv
Serving fix. Editing the middle of the context breaks the normal prefix KV cache. So they built Suffix Cache Reuse in SGLang, which reuses cached KV for unchanged text after an edit and re-rotates RoPE positions. It cuts server-side compute by about 35% at matched accuracy. 
arxiv
Paper 2: Context-Folding / FoldGRPO, arXiv 2510.11967 (ByteDance Seed / CMU, Oct 2025)

Novelty, in one line: Let the agent “branch” into a side-task and then “fold” it away, keeping only a short result. Train this behavior end-to-end with RL.

Method, simply:

Two tools. branch(description, prompt) opens a sub-task. return(message) closes it. On return, everything done inside the branch is removed from context, and only the return message remains. It’s like a function call whose local variables get thrown away. 
arxiv
KV-cache friendly. Return just rolls the cache back to the branch point, so the prefix stays intact.
FoldGRPO. This is GRPO trained on the folded context, plus dense token-level penalties:
an unfolded-token penalty when the main thread grows past 50% of the budget without branching,
an out-of-scope penalty, where an LLM judge checks whether the branch stayed on its sub-task,
a failure penalty for broken tool calls.
Results. With a 32K active context and up to 10 branches, Seed-OSS-36B reaches 62.0% on BrowseComp-Plus and 58.0% on SWE-Bench Verified. That beats a 327K-context ReAct agent. 
arxiv
The core difference in code
python
# Standard agent: context only grows
ctx = ctx + model(ctx)

# Context-Folding: a fixed, stack-shaped edit
#   branch -> push a marker;  return -> pop back to the marker, keep a summary
ctx = ctx[:branch_start] + [return_msg]

# CLM: any edit the model writes itself
ctx = model_writes_new_ctx(ctx)   # delete, rewrite, insert notes, offload to disk...

Folding is one specific choice of how to edit the context. CLM lets the model decide.

Comparison
	Context-Folding (2025)	CLM (2026)
What the model controls	2 fixed actions (branch/return)	The whole context, via Bash
Edit shape	Stack-like: can only collapse a finished branch	Anywhere: middle edits, rewrites, offloading
Needs training?	Yes; gains come mainly from FoldGRPO	No; works zero-shot, RL and prompting are optional extras
RL reward design	Hand-designed process penalties (unfolded tokens, scope judge)	Outcome reward + FLOPs bonus, success-gated only
KV cache	Naturally prefix-friendly	Breaks the prefix cache, so needs Suffix Cache Reuse
Philosophy	Structured, learnable summarization at subtask boundaries	“Bitter lesson”: don’t pre-define the strategy, let it emerge
Weak spot	Can’t remove content already sitting in the main thread	More freedom means more ways to fail; models can write injected instructions into their own context

Where Folding loses, per CLM’s own test: In CLM’s ContextBench, a branch starts from a copy of a context that already holds the data and cannot delete anything from the main thread, so the main thread stays just as full. When large inputs are pushed straight into the main context, folding has no way to evict them. 
arxiv

Where Folding still looks good:

It’s more predictable.
It’s cache-efficient for free.
Its process rewards give a denser learning signal, which matters more for smaller models. CLM admits that before RL, the 9B Qwen CLM trailed the summary harness by six points. 
arxiv

On the headline numbers: don’t compare 62.0% vs 59.4% on BrowseComp-Plus directly.

Folding trained on part of BrowseComp-Plus and tested on a 150-question split.
CLM tested on all 830 questions, with no training (27B) or with training on a different dataset (9B).

Why it matters for post-training: both papers broadcast a trajectory-level advantage across non-append-only trajectories, meaning each segment is trained under the context it was actually generated in. The papers differ in how they shape the reward:

Folding uses dense process penalties.
CLM ranks successful trajectories by efficiency.

Your rubric/RLVR work would fit naturally as a third option there.
# Todo

- [x] 2026-11-03 Read proposal.md, notes.md, lit_survey.md
- [x] 2026-11-03 Write docs/proposal_v2.md (scoped to mentor feedback)
- [x] 2026-11-08 Read CLM repo @4831982, MEM1 repo, RL framework survey
- [x] 2026-11-08 Write docs/implementation_plan.md
- [x] 2026-11-10 Update proposal_v2.md for round-2 notes and teacher-only reward
- [x] 2026-11-10 Revise implementation_plan.md: reproduction first, teacher-only arm, new gates
- [x] 2026-11-10 User: "start with M0 to M2"

## M0 setup (done 2026-11-10)
- [x] pyproject (py3.12, uv), pinned CLM @4831982, ContextBench @fb533c1 checkout in third_party/
- [x] Modal volumes (teachclm-hf-cache, teachclm-vllm-cache, teachclm-wiki18) + secret teachclm-api-key (local copy in .env)
- [x] wiki-18 e5 index + corpus on teachclm-wiki18: 21,015,324 passages, corpus lines verified against index rows
- [x] SkyRL step-wise + LoRA smoke test: PASS on Qwen3.5-0.8B, 1x L40S (infra/skyrl_smoke.py).
      Not exercised: prompts that rewrite earlier context; needs our own generator (M8).

## M1 serving (done 2026-11-10)
- [x] infra/serve_vllm.py deployed (app teachclm-vllm, H100, scales to zero): tool call round-trips
- [~] infra/retriever.py deployed (app teachclm-retriever, 16 CPU / 80 GB, scales to zero): results correct;
      warm latency 0.76-1.02 s measured from the laptop (target < 1 s). 32 CPUs did not help.
      Searches are serialized, so move to a GPU index if parallel evals in M4 are slowed by it.

## M2 reproduce paper zero-shot (pipeline done; target needs a user decision)
- [x] Upstream ClmAgent (ClmStream, unmodified) via Harbor + Colima against our served 9B: works.
      Results in runs/m2/: kv-store R130 1.00; slice1 (2 seeds each) kv-store R1000 1.00/1.00,
      needle K4-C24 1.00/1.00, sudoku t30 1.00/1.00, log-triage L1600 0.67/0.00 (one budget_wall).
- [ ] FINDING: the paper reports no Qwen3.5-9B number on ContextBench (its figure is GPT-5.4; ICL uses
      Qwen3.6-27B). The paper's 9B zero-shot numbers are on BrowseComp-Plus (CLM 39.9% vs summary 37.7%,
      830 questions; second-hand from a summary of the paper HTML, check against the PDF).
      User to decide: BrowseComp-Plus subset as the reproduction target, or accept ContextBench as a
      pipeline check only.

## Later
- [ ] Still needed from user: Modal budget, course deadline, include hybrid arm (H) or not
- [ ] M3 loop+parity → M4 task+gates → M5 baselines → M6 prompt evolution → M7 edits/oracle audit/teacher → M8 RL plumbing → M9 RL arms → M10 behaviors

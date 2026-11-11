"""M0 smoke test: does SkyRL step-wise GRPO + LoRA train Qwen3.5 on one Modal GPU?

Run (ephemeral, never deploy):

    uvx modal run infra/skyrl_smoke.py
    uvx modal run infra/skyrl_smoke.py --model Qwen/Qwen3.5-2B --steps 3

What it does: installs SkyRL at a pinned commit, builds the GSM8K multi-turn
dataset that ships with SkyRL (examples/train/turn_level_rewards), and runs a
tiny colocated FSDP + vLLM GRPO job with `generator.step_wise_trajectories=true`
and a rank-8 LoRA. PASS = >=2 optimizer steps, finite loss, and more training
sequences than trajectories (i.e. trajectories were split into per-step samples).
"""

import modal

SKYRL_REPO = "https://github.com/NovaSky-AI/SkyRL.git"
SKYRL_COMMIT = "8adc1e53465e2766ac8e8f5ec652e633d17e2279"  # tag skyrl-v0.4.0
# SkyRL's own base image (examples/train_integrations/modal/main.py), pinned by digest.
BASE_IMAGE = (
    "novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0"
    "@sha256:609b7df001fa41a657d1d6a3607cb78af9290208d2f3397e4c7ce7182ec1d622"
)
REPO_DIR = "/root/SkyRL"
PY = f"{REPO_DIR}/.venv/bin/python"
GPU = "L40S"

image = (
    modal.Image.from_registry(BASE_IMAGE)
    .run_commands(
        f"git clone {SKYRL_REPO} {REPO_DIR} && cd {REPO_DIR} && git checkout {SKYRL_COMMIT}",
        # uv.lock pins everything (vllm 0.30.0, torch 2.13.0, peft 0.18.1, ray 2.57.0, ...)
        f"cd {REPO_DIR} && uv sync --frozen --extra fsdp --python 3.12",
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("teachclm-skyrl-smoke")
hf_cache = modal.Volume.from_name("teachclm-skyrl-hf-cache", create_if_missing=True)

# Trajectories per training step = TRAIN_BATCH_SIZE * N_SAMPLES.
TRAIN_BATCH_SIZE = 4
N_SAMPLES = 4
MAX_TURNS = 3


def _overrides(model: str, steps: int, data_dir: str) -> list[str]:
    return [
        f"data.train_data=['{data_dir}/train.parquet']",
        f"data.val_data=['{data_dir}/validation.parquet']",
        "environment.env_class=gsm8k_multi_turn",
        # --- the three things under test ---
        "generator.step_wise_trajectories=true",
        "trainer.algorithm.advantage_estimator=grpo",
        "trainer.policy.model.lora.rank=8",
        "trainer.policy.model.lora.alpha=16",
        # --- model (Qwen3.5 flags from examples/train/models/run_qwen3.5_0.8b.sh) ---
        f"trainer.policy.model.path={model}",
        "trainer.policy.language_model_only=true",
        "generator.inference_engine.language_model_only=true",
        "trainer.remove_microbatch_padding=false",
        "generator.chat_template_kwargs.enable_thinking=false",
        # --- single GPU, colocated FSDP trainer + local vLLM engine ---
        "trainer.strategy=fsdp",
        "trainer.placement.colocate_all=true",
        "trainer.placement.policy_num_gpus_per_node=1",
        "trainer.placement.ref_num_gpus_per_node=1",
        "generator.inference_engine.backend=vllm",
        "generator.inference_engine.run_engines_locally=true",
        "generator.inference_engine.num_engines=1",
        "generator.inference_engine.tensor_parallel_size=1",
        "generator.inference_engine.weight_sync_backend=nccl",
        "generator.inference_engine.gpu_memory_utilization=0.5",
        # --- tiny job ---
        f"trainer.max_training_steps={steps}",
        "trainer.epochs=1",
        f"trainer.train_batch_size={TRAIN_BATCH_SIZE}",
        f"trainer.policy_mini_batch_size={TRAIN_BATCH_SIZE}",  # => 1 optimizer step per training step
        "trainer.micro_forward_batch_size_per_gpu=8",
        "trainer.micro_train_batch_size_per_gpu=4",
        "trainer.update_epochs_per_batch=1",
        f"generator.n_samples_per_prompt={N_SAMPLES}",
        f"generator.max_turns={MAX_TURNS}",
        "generator.batched=false",
        "generator.use_conversation_multi_turn=true",
        "trainer.max_prompt_length=512",
        "generator.max_input_length=2048",
        "generator.sampling_params.max_generate_length=256",
        "trainer.policy.optimizer_config.lr=1.0e-5",
        "trainer.algorithm.use_kl_loss=false",
        # --- no eval / checkpoints / wandb ---
        "trainer.eval_before_train=false",
        "trainer.eval_interval=-1",
        "trainer.ckpt_interval=-1",
        "trainer.hf_save_interval=-1",
        "trainer.resume_mode=null",
        "trainer.logger=console",
        "trainer.project_name=teachclm-skyrl-smoke",
        "trainer.run_name=stepwise-lora",
        "trainer.log_path=/tmp/skyrl-logs",
        "trainer.ckpt_path=/tmp/skyrl-ckpts",
        "trainer.export_path=/tmp/skyrl-exports",
    ]


@app.function(
    image=image,
    gpu=GPU,
    timeout=40 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    max_containers=1,
)
def smoke(model: str, steps: int) -> dict:
    import glob
    import os
    import re
    import subprocess
    import time

    os.chdir(REPO_DIR)
    data_dir = "/tmp/data/gsm8k_multi_turn"
    env = {**os.environ, "SKYRL_DUMP_INFRA_LOG_TO_STDOUT": "1"}  # vLLM/worker logs on stdout

    subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, check=True)
    subprocess.run(
        [PY, "examples/train/turn_level_rewards/gsm8k_multi_turn_dataset.py",
         "--output_dir", data_dir, "--max_turns", str(MAX_TURNS)],
        check=True,
    )

    cmd = [PY, "-m", "skyrl.train.entrypoints.main_base", *_overrides(model, steps, data_dir)]
    print("RUN:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    rc = proc.wait()
    wall = time.time() - t0
    log = "".join(lines)

    # If infra logs went to files instead of stdout, show their tails on failure.
    if rc != 0:
        for path in glob.glob("/tmp/skyrl-logs/*.log"):
            print(f"===== tail {path} =====")
            subprocess.run(["tail", "-n", "200", path])

    num = r"(-?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))"
    batch_num_seq = [int(x) for x in re.findall(r"batch_num_seq: (\d+)", log)]
    losses = [float(x) for x in re.findall(r"policy_loss['\"]?[:=]\s*" + num, log, flags=re.I)]
    n_traj = TRAIN_BATCH_SIZE * N_SAMPLES
    finite = bool(losses) and all(x == x and abs(x) != float("inf") for x in losses)
    result = {
        "returncode": rc,
        "train_wall_s": round(wall, 1),
        "trajectories_per_step": n_traj,
        "batch_num_seq_per_step": batch_num_seq,
        "policy_losses": losses,
        "training_done": "Training done!" in log,
        "split_into_steps": bool(batch_num_seq) and all(n > n_traj for n in batch_num_seq),
        "loss_finite": finite,
    }
    result["PASS"] = (
        rc == 0 and len(batch_num_seq) >= 2 and len(losses) >= 2
        and result["split_into_steps"] and finite and result["training_done"]
    )
    print("SMOKE_RESULT", result, flush=True)
    return result


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3.5-0.8B", steps: int = 3):
    result = smoke.remote(model, steps)
    print("SMOKE_RESULT", result)
    print("PASS" if result["PASS"] else "FAIL")

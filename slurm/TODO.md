# NERSC Perlmutter Launch — TODO

Checklist to get `slurm/opd_train.sbatch` running on Perlmutter from a fresh
NERSC account. Submit with `sbatch slurm/opd_train.sbatch` once everything
below is green.

## 0. NERSC essentials — node types & where to do what

Perlmutter has four distinct node classes. Knowing which one you're on
determines what you're *allowed* to do.

| Node class                    | Count | CPU                       | GPU                      | Memory | Local SSD          |
| ----------------------------- | ----- | ------------------------- | ------------------------ | ------ | ------------------ |
| **Login**               | 40    | 2× AMD EPYC 7713 (Milan) | 1× A100 40 GB (shared!) | 512 GB | 960 GB             |
| **GPU compute (40 GB)** | 1,536 | 1× EPYC 7763, 64 cores   | 4× A100 40 GB           | 256 GB | —                 |
| **GPU compute (80 GB)** | 256   | 1× EPYC 7763, 64 cores   | 4× A100 80 GB           | 256 GB | —                 |
| **CPU compute**         | 3,072 | 2× EPYC 7763 (128 cores) | —                       | 512 GB | —                 |
| **Data-transfer (DTN)** | —    | —                        | —                       | —     | high-bandwidth WAN |

### Where `ssh <user>@perlmutter.nersc.gov` puts you

A **login node** (random, behind a load balancer). They each have one A100
40 GB attached — useful for `import torch; torch.cuda.is_available()`-class
sanity checks, but **shared with every other logged-in user**. Do not run
training, vLLM, or anything multi-process there: NERSC's login-node watchdog
will kill long-running CPU/GPU processes.

### Where to do what

| Task                                                                                                                        | Node                              | How                                                                                                                                                                                                                                                                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Edit code,`git`, `conda create`, `pip install`, `huggingface-cli download`                                          | **Login**                   | `ssh perlmutter.nersc.gov`. Login nodes have outbound internet — required, since compute nodes don't.                                                                                                                                                                                                                                                    |
| Large bulk transfers (datasets, checkpoints to/from `$CFS`)                                                               | **DTN**                     | `ssh dtn01.nersc.gov` or use Globus. Don't push tens of GB through login nodes.                                                                                                                                                                                                                                                                           |
| **Test the OPD code** end-to-end on real hardware (does verl init? does ray start? does the first training step run?) | **Interactive GPU compute** | `salloc --nodes 1 --qos interactive --time 2:00:00 --constraint 'gpu&hbm80g' --gpus 4 --account m4788_g` — gives a real 4-GPU node in ~6 min. Once it returns you have a shell on the compute node and can run `bash slurm/opd_train.sbatch` (or invoke `on_policy_distillation.sh` directly) interactively. Kill it after a few steps look healthy. |
| **Submit the production job**                                                                                         | From a**login node**        | `sbatch slurm/opd_train.sbatch`. The scheduler then runs the sbatch on a GPU compute node.                                                                                                                                                                                                                                                                |

### Things that only work on login / DTN nodes (not compute)

- Outbound internet → `pip install`, `huggingface-cli download`, `wandb sync`,
  `git pull` from github. Compute nodes are firewalled, so we set
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `WANDB_MODE=offline` in the
  sbatch and sync afterwards.
- `sbatch` / `salloc` / `squeue` itself (Slurm client lives on login nodes).

### Recommended workflow

1. **Login node** → set up env, pre-fetch HF models, sync small files, edit code.
2. **DTN** → big rsyncs (datasets, checkpoint backups).
3. **Interactive GPU compute** (`salloc ... --qos interactive`) → smoke-test
   the sbatch's pipeline on real 4× A100 hardware before burning hours in
   the queue. Run a few training steps, confirm vLLM init / ray cluster /
   reward model load all succeed, then `exit`.
4. **Login node** → `sbatch slurm/opd_train.sbatch` for the real run.
   Monitor via `squeue --me` and `tail -f $DATA_ROOT/logs/slurm-*-<jobid>.out`.

### Requirements

all env, data, huggingface, ckpt saving, etc. goes to /pscratch/sd/y/yequan, not home directory

## 1. Account & paths

- [ ] Confirm Slurm account string. The template uses `--account=m4788_g`
  (inherited from the DoRA example); replace with your own project if
  different. Check via `iris` or `sacctmgr show user $USER`.
- [ ] Pick `OPD_REPO` (typically `$HOME/OPD` or a `$CFS/<project>/OPD` clone)
  and `DATA_ROOT` (always under `$PSCRATCH/<your-user>/opd`). Edit the
  two `export ... :- ...` defaults near the top of `opd_train.sbatch`.
- [ ] `mkdir -p $DATA_ROOT/{logs,checkpoints,huggingface,wandb,envs}` on
  a login node before the first submission. Slurm `--output=` won't
  auto-create the parent directory.

> **Scratch purge.** `$PSCRATCH` is purged after ~8 weeks idle. Copy final
> checkpoints to `$CFS` or `$HOME` (40 GiB quota — checkpoints don't fit).

## 2. Conda env (`verl`, py3.12)

Install on a login node (compute nodes have no internet):

- [ ] Install Miniforge under `$DATA_ROOT/envs/miniforge3` if not already.
- [ ] `conda create -p $DATA_ROOT/envs/verl python=3.12 -y`
- [ ] `conda activate $DATA_ROOT/envs/verl`
- [ ] `cd $OPD_REPO/verl && USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh`
- [ ] `pip install math-verify`
- [ ] Smoke test: `python -c "import torch, vllm, ray, verl; print('ok')"`
- [ ] Smoke test GPU: grab an interactive node and `python -c "import torch; print(torch.cuda.device_count())"` — expect 4.

## 3. Pre-fetch HuggingFace assets

Compute nodes are offline (`HF_HUB_OFFLINE=1` is set in the sbatch). Prime
the cache on a login node:

- [ ] `export HF_HOME=$DATA_ROOT/huggingface`
- [ ] `huggingface-cli download Qwen/Qwen3-1.7B`
- [ ] `huggingface-cli download Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`
- [ ] (If swapping `ACTOR_MODEL_PATH` / `REWARD_MODEL_PATH`) pre-download
  whatever else you plan to point at.

## 4. Datasets

- [ ] Sync the `datasets/` tree to `$OPD_REPO/datasets/` (rsync from this
  dev box, or re-download via the recipes in `scripts/infer/`).
- [ ] Verify: `ls $OPD_REPO/datasets/dapo-math-17k.parquet` exists.
- [ ] Verify: `ls $OPD_REPO/datasets/test_data/{AIME24,AIME25,AMC23}/test.parquet` all exist.
  (These three are what the launcher's default `TEST_DATASET` references.)

## 5. Sanity check the sbatch

- [ ] `cat slurm/opd_train.sbatch` and confirm the placeholder paths
  (`/global/homes/l/liyantan/OPD`, `/pscratch/sd/l/liyantan/opd`) have
  been replaced with yours, OR pass overrides via
  `sbatch --export=ALL,OPD_REPO=...,DATA_ROOT=... slurm/opd_train.sbatch`.
- [ ] Dry-run check the script syntax: `bash -n slurm/opd_train.sbatch`.
- [ ] Confirm QoS / wallclock fit your project budget: `regular` allows up
  to 48 h on Perlmutter GPU; the template requests 24 h. Switch to
  `--qos=preempt` (25% rate, may be requeued) or `--qos=debug` (30 min
  max, ≤8 nodes) for cheap shakeout runs.

## 6. First submission

- [ ] `sbatch slurm/opd_train.sbatch`
- [ ] `squeue --me`
- [ ] `tail -f $DATA_ROOT/logs/slurm-opd_train-<jobid>.out`
- [ ] Watch first ~5 min for: ray head started, vllm engine init, first
  generation batch completes. If it dies in vllm init, the usual
  culprits are FlashInfer (already disabled) or wrong CUDA toolkit
  (`module load cudatoolkit/12.4` may need bumping if torch is newer).

## 7. After the job

- [ ] Sync wandb metrics on a login node (compute nodes can't reach the
  internet):
  `wandb sync $DATA_ROOT/wandb/offline-run-*`
- [ ] Copy final checkpoint out of `$PSCRATCH` before purge:
  `rsync -a $DATA_ROOT/checkpoints/<run>/ $CFS/<project>/checkpoints/<run>/`

## Open questions / nice-to-haves

- [ ] **Multi-node (2 nodes × 4 GPUs = 8 GPUs)** to match the paper's
  8× A800 80GB setup. Requires:
  - `--nodes=2`, change `NNODES=2`.
  - Replace `ray start --head` (inside `on_policy_distillation.sh`)
    with a head-vs-worker `srun` pattern. Probably cleanest to keep
    the multi-node logic *outside* `on_policy_distillation.sh` —
    write a `slurm/opd_train_2node.sbatch` that starts ray itself
    and then calls a stripped-down launcher.
- [ ] **40 GB fallback (`--constraint=gpu`)** — would let the job land on
  more nodes / queue faster. Needs `ACTOR_PARAM_OFFLOAD=True` and
  lower `GPU_MEMORY_UTILIZATION` (~0.6). Untested.
- [ ] **Reservation / preempt-friendly checkpoints** — verl saves every
  `SAVE_FREQ=20` steps; confirm that's frequent enough that a
  preempt-QoS requeue loses < 1 h of work.

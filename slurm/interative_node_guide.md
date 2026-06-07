# Long runs on NERSC interactive nodes with auto-relaunch after 4h expiry

How to run a multi-hour (multi-node) training job on Perlmutter **interactive** GPU
nodes and have it survive the **4-hour interactive walltime cap** by checkpointing
and auto-resuming. Distilled from the working `zo_opd_qwen4b_1p7b` 2-node run.

## Why interactive nodes at all?

The `regular` GPU queue can sit pending for many hours (we saw ~21 h for a full-node
24 h request). Interactive nodes schedule in **minutes** but are capped at **4 h** and
**4 nodes**. The trick: grab interactive nodes, checkpoint frequently, and have a
controller automatically re-grab nodes and resume from the last checkpoint each time
the 4 h runs out. Net effect: a long run completes in fast-scheduling 4 h chunks.

Trade-off: each 4 h boundary costs a re-queue + re-init (~5-10 min) and loses at most
`SAVE_FREQ` steps of work. For a job that finishes in a few hours this beats the batch
queue; for very long jobs, weigh it against a single `regular`/`preempt` batch job.

## The four moving parts

```
controller (login node, detached)         slurm/opd/full/zo_opd_job1_full_controller.sh
  └─ loops: salloc -N2 --qos interactive --time 4:00:00  bash inside.sh
        inside.sh (login node, under alloc)  slurm/opd/zo_opd_2node_inside.sh
          ├─ ray start --head   on node0   (srun --block &)
          ├─ ray start --address on node1  (srun --block &)
          └─ driver on node0 (srun, RAY_EXTERNAL=1) ── env.sh ── on_policy_distillation.sh
                env.sh + rayenv.sh                    slurm/opd/full/zo_opd_2node_env.sh + slurm/opd/zo_opd_2node_rayenv.sh
```

- **controller** — the only long-lived process. `salloc ... bash inside.sh` blocks
  until the driver finishes OR the 4 h limit revokes the allocation (`rc=143`). On
  revoke it loops and re-allocates. Stops when the checkpoint reaches the final step
  (`rc=0`) or when no checkpoint progress is made across several allocations (stall).
- **inside.sh** — bootstraps a Ray cluster across the allocation's nodes, then runs the
  driver on the head node attached to that cluster.
- **env.sh** — the driver's env: hyperparameters + a **fixed** checkpoint dir, then calls
  the normal launcher.
- **rayenv.sh** — the runtime/system env that **every Ray daemon** sources (see gotcha #3).

`salloc <opts> <command>` runs `<command>` on the **login node** (not the compute node)
with `SLURM_JOB_ID`/`SLURM_JOB_NODELIST` set; the command uses `srun` to place work on
the compute nodes. That's why `inside.sh` can drive everything from one process.

## What makes resume work

1. **Fixed checkpoint dir + experiment name.** The stock launcher stamps `$(date)` into
   `CKPT_PATH`/`EXPERIMENT_NAME`, so every relaunch would get a *new* dir and never
   resume. Pin them (we made `on_policy_distillation.sh` honor `${CKPT_PATH:-…}` /
   `${EXPERIMENT_NAME:-…}` env overrides).
2. **`trainer.resume_mode=auto`** (verl default) — reads `latest_checkpointed_iteration.txt`
   in `default_local_dir` and resumes (loads model+optimizer+lr_scheduler+rng on every rank).
3. **Save often.** `SAVE_FREQ` small enough that a mid-segment expiry loses little
   (we used 10 → ≤~13 min lost).
4. **Keep only the latest checkpoint.** `trainer.max_actor_ckpt_to_keep=1` only rotates
   checkpoints saved *within one process*, so the checkpoint a segment **resumes from**
   lingers after the next save. A background pruner in the controller deletes any
   `global_step_*` older than `latest_checkpointed_iteration.txt` (never the latest or an
   in-progress higher-numbered save → safe to run continuously).
5. **One wandb run across segments.** Pin `WANDB_RUN_ID` + `WANDB_RESUME=allow` so all
   segments append to a single offline run; `wandb sync` consolidates them online.

## Multi-node gotchas (each cost a debug cycle)

1. **Conda in every `srun` step.** Compute-node `srun` steps don't get a login shell, so
   `ray`/`python` aren't on PATH (`ray: No such file or directory`). Source conda inside
   each step.
2. **`srun ... env VAR=x cmd` can break.** `env` PATH-resolved to a broken
   `~/.local/bin/env` (`Permission denied`). Use `srun bash -c "export VAR=x; exec cmd"`.
3. **Ray actors inherit the RAYLET's env, not the driver's.** This is the big one. The
   driver's exported env (HF, wandb, vLLM flags, NCCL, caches) does **not** reach the
   actors that do the real work — they inherit from the `ray start` step. So all of that
   env must be on the `ray start` steps (we put it in `rayenv.sh`, sourced by every step).
   This is why a single-node run "just works" (ray started in a fully-exported shell) but
   the multi-node version failed until the env was moved onto the daemons.
4. **`OSError [Errno 524]` = `flock` failing on GPFS/Lustre (ENOTSUPP).** HuggingFace and
   JIT compilers lock cache files. Fixes: keep the HF cache on `$PSCRATCH` (flock OK),
   put JIT caches (triton/inductor/flashinfer/outlines, `XDG_CACHE_HOME`) on **node-local
   `/tmp`**, and disable the flashinfer sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`) — its
   JIT compiles + flocks on first use. Never let any cache default to `$HOME` (GPFS).
5. **`CUDA_VISIBLE_DEVICES` default.** `on_policy_distillation.sh` defaults it to `6,7`;
   pin `0,1,2,3` for a 4-GPU node.
6. **Keep Ray daemons alive** with `ray start … --block &` (a backgrounded srun step).
   Without `--block` the daemon dies when the step ends. Run the driver as a separate
   `srun --overlap` step on the head node so it can share the node with the head daemon.
7. **`ray.init()` attaches to the external cluster** when `RAY_ADDRESS` is set. We gated
   the launcher with `RAY_EXTERNAL=1` to skip its own `ray stop`/`ray start --head`.

## Launch / monitor / stop

```bash
# LAUNCH (from a login node) — detached, with a log:
nohup bash slurm/opd/full/zo_opd_job1_full_controller.sh \
  > /pscratch/sd/$USER/opd/logs/controller_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# MONITOR
tail -f /pscratch/sd/$USER/opd/logs/controller_*.log     # attempts / resume points / completion
squeue --me                                              # current allocation + 4h countdown (%L)
tail -f /pscratch/sd/$USER/opd/logs/zo2node_attempt*.log | grep global_step   # live steps
ls /pscratch/sd/$USER/opd/checkpoints/<run>/             # should always be ONE global_step_N

# STOP (must do BOTH, or the controller just re-allocates):
scancel <jobid>
pkill -f <controller-name>   # e.g. zo_opd_lora_controller, zo_opd_job1_full_controller

# AFTER (login node): publish results, copy checkpoint off the purge-prone scratch
wandb sync /pscratch/sd/$USER/opd/wandb/wandb/offline-run-*<run-id>
rsync -a /pscratch/sd/$USER/opd/checkpoints/<run>/global_step_<final>  $CFS/<project>/
```

Controller log reads like: `attempt 1 done rc=143 | iter 0->130` (expiry) → `attempt 2
| start_iter=130` (resume) → … → `TRAINING COMPLETE: iter N >= total N` (`rc=0`).

## Adapting to a different run

- **Node count:** change `--nodes` in the controller's `salloc` and `NNODES` in `env.sh`
  (`inside.sh` derives head/workers from `$SLURM_JOB_NODELIST` automatically). Stay ≤ 4
  (interactive cap).
- **Models / dataset / hyperparameters:** edit `env.sh` (or pass overrides via the
  controller's environment — every knob uses `${VAR:-default}`).
- **Different trainer entirely:** keep the controller + inside.sh skeleton and the five
  fixes above; swap `env.sh` to launch your script with a fixed checkpoint dir and
  `resume_mode=auto` (or your framework's equivalent).
- **Stall safety:** the controller stops after `STALL_LIMIT` (default 3) allocations with
  no checkpoint progress — i.e. a real failure, not just an expiry. Check the named
  `runlog` when that happens.

## Reference implementation

the `slurm/opd/{<recipe>/controller+env, zo_opd_2node_inside, zo_opd_2node_rayenv}.sh` scripts in this repo, plus the
`RAY_EXTERNAL` / overridable-`CKPT_PATH` patches in `on_policy_distillation.sh`.
Verified end-to-end: a 279-step run completed across **three** 4 h expiries, each
auto-resuming from the last checkpoint, with a single checkpoint kept throughout.

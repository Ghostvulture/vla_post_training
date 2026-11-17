# RoboDojo experiment jobs

These scripts capture the X-VLA screening, success-filtered fine-tuning, RL, module toggle, learning-rate comparison, second collection seed and checkpoint repeat runs. The follow-up jobs refer to historical Slurm IDs, adapter directories and layout views. When replaying the sequence in a fresh workspace, substitute the matching new run IDs and paths. The learning-rate recovery job reuses the first 25 episodes of job 20070.

On the server export FYP_STORAGE, FYP_RUNTIME and FYP_COMPUTE_NODE (the allocated compute node). Submit to that node through Slurm, excluding the login node. For example, after exporting the three variables, submit the RL training job with

```bash
sbatch --partition=GPU --nodelist="$FYP_COMPUTE_NODE" --exclude=4090node1 --gres=gpu:4090:1 --cpus-per-task=8 --ntasks=1 --time=05:00:00 evaluations/fyp/robodojo/slurm/robodojo-rl.sbatch
```

For local report analysis, run either script in the adjacent analysis directory with the report output directory as its positional argument. Its parent must contain the archived robodojo-* run directories with their results.json files.

## Two-round supplemental pilot

Use `robodojo-rounds.sbatch train1` to train RL–Shared (2e-6) and native FSFT (1e-4) on the shared job-20010 collection. Both save steps 0/8/16, Adam state and RNG, audit continuation by repeating an actual update, and validate the behavior replay or action conversion. FSFT exports merged weights separately from resumable adapters.

After training passes, use phases `eval0` and `eval16 <train1-result-directory>` for the two seeded baseline paths and first-round checkpoints. Use `round2 <train1-result-directory>` to collect 20 new episodes per method, continue each optimizer for 16 updates, and evaluate step 32. A-Train uses seed 0 IDs 0–19; A-Val/B-development use seed 1 IDs 10–29 with the historical 25+5 batching. Collection seed 43 and evaluation seed 1701 are fixed. All-equal RL rewards or empty complete-success FSFT data skip that round without additional collection. `COMPLETE` is written only after every phase operation succeeds.

Run the CPU data-contract checks with `PYTHONPATH=evaluations/fyp/robodojo python -m unittest discover -s evaluations/fyp/robodojo -p test_round_data.py`. Model replay, export and optimizer checks require the server GPU.

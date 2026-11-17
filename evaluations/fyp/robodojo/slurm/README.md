# RoboDojo experiment jobs

These scripts capture the X-VLA screening, success-filtered fine-tuning, RL, module toggle, learning-rate comparison, second collection seed and checkpoint repeat runs. The follow-up jobs refer to historical Slurm IDs, adapter directories and layout views. When replaying the sequence in a fresh workspace, substitute the matching new run IDs and paths. The learning-rate recovery job reuses the first 25 episodes of job 20070.

On the server export FYP_STORAGE, FYP_RUNTIME and FYP_COMPUTE_NODE (the allocated compute node). Submit to that node through Slurm, excluding the login node. For example, after exporting the three variables, submit the RL training job with

```bash
sbatch --partition=GPU --nodelist="$FYP_COMPUTE_NODE" --exclude=4090node1 --gres=gpu:4090:1 --cpus-per-task=8 --ntasks=1 --time=05:00:00 evaluations/fyp/robodojo/slurm/robodojo-rl.sbatch
```

For local report analysis, run either script in the adjacent analysis directory with the report output directory as its positional argument. Its parent must contain the archived robodojo-* run directories with their results.json files.

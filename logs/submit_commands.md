```
jjozefowicz@eden:~/git/bml-big-2$ sbatch submit_begin.sub
Submitted batch job 977270
jjozefowicz@eden:~/git/bml-big-2$ sbatch submit_torchrun.sub
Submitted batch job 977271
jjozefowicz@eden:~/git/bml-big-2$ sbatch submit_2gpu.sub
Submitted batch job 977272
jjozefowicz@eden:~/git/bml-big-2$ sbatch sbatch_save.sub
Submitted batch job 977273
jjozefowicz@eden:~/git/bml-big-2$ sbatch -d afterok:977273 sbatch_load.sub
Submitted batch job 977274
jjozefowicz@eden:~/git/bml-big-2$ sbatch sbatch_grid_search.sub
Submitted batch job 977300
```
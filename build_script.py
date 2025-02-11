import argparse

parser = argparse.ArgumentParser()
#todo add all params here
# numnodes
# numgpus per node
# https://docs.nvidia.com/dgx-cloud/slurm/latest/cluster-user-guide.html
# https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/slurm/sbatch_run.sh


--partition=plgrid-gpu-a100

--account=plgllmparamgr-gpu-a100


`/mnt/evafs/software/slurm/amd_epyc/current/bin` <- dgx, sr, hopper
oraz 
`/mnt/evafs/software/slurm/intel_broadwell/current/bin` <- pascal
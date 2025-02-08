#!/bin/bash

source data/export.env

torchrun \
--standalone \
--nnodes=1 \
--nproc-per-node=1 \
main.py -bs 64 --no-is_plgrid --load_dir model-save -n 1000
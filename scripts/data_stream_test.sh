#!/bin/bash

module load StdEnv/2023
module load gcc/12.3 arrow/16.1.0 python/3.11.5 hdf5/1.14.2
module load httpproxy
source /home/heesters/projects/def-sfabbro/heesters/envs/ssl_env/bin/activate

python /home/heesters/projects/def-sfabbro/heesters/github/TileSlicer/data_stream_test.py
With GPU:

sbatch --time=00:20:00 --cpus-per-task=6 --account=rrg-kyi --mem=100000M --gres=gpu:1 ./scripts/mim_unions.sh

Without GPU:

sbatch --time=00:20:00 --cpus-per-task=6 --account=def-sfabbro --mem=100000M ./scripts/data_stream_test.sh
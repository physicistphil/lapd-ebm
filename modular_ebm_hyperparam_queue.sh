#!/bin/bash

# size=$1
# port=$2

# if [[ "$size" == "small" ]]; then
# 	width=64
# 	head=2
# elif [[ "$size" == "medium" ]]; then
# 	width=128
# 	head=4
# elif [[ "$size" == "large" ]]; then
# 	width=256
# 	head=8
# elif [[ "$size" == "xlarge" ]]; then
# 	width=512
# 	head=16
# else
# 	echo "choose between small, medium, large, and xlarge"
# 	exit 1
# fi

# run_script() {
# 	$(python modular_ebm.py --port $port --time_limit 4 \
#   --embed_dim $width \
#   --dense_width $width \
#   --num_heads $head \
#   --num_hidden $width \
#   --num_msi_attn 1 \
#   --energy_num_heads $head \
#   --energy_num_hidden $width \
#   --energy_num_attn 4 \
#   --dataset $dataset > /dev/tty)
# }


# dataset=data-MSI-mini_2022-9-28_sets-4-train.npz
# run_script

# dataset="data-MSI-mini_2022-9-28_sets-3-train.npz"
# run_script

# dataset="data-MSI-mini_2022-9-28_sets-2-train.npz"
# run_script

# dataset="data-MSI-mini_2022-9-28_sets-1-train.npz"
# run_script

python modular_ebm.py --time_limit 3 --lr 1e-2 --sample_steps 1000 --step_size 0.001

python modular_ebm.py --time_limit 3 --lr 1e-3 --sample_steps 1000 --step_size 0.001

python modular_ebm.py --time_limit 3 --lr 1e-4 --sample_steps 1000 --step_size 0.001
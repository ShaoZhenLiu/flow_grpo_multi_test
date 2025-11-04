export PYTHONPATH=../../:$PYTHONPATH

set -e
set -x
set -o pipefail

# 1 GPU
export CUDA_VISIBLE_DEVICES=1

torchrun \
    --standalone \
    --nproc_per_node=1 \
    --master_port=19501 \
    scripts/train_omni_gen.py \
    --config config/grpo.py:multi_human_omni_gen_1gpu

# 2 GPU
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# torchrun \
#     --standalone \
#     --nproc_per_node=4 \
#     --master_port=19501 \
#     scripts/train_omni_gen.py \
#     --config config/grpo.py:multi_human_omni_gen_1gpu


# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu

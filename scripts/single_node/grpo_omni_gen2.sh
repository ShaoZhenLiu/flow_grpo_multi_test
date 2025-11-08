# # 1 GPU
set -x
export CUDA_VISIBLE_DEVICES=1

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes=1 \
    --main_process_port 34501 \
    scripts/train_omni_gen2.py \
    --config config/grpo.py:multi_omni_gen2_1gpu


# 2 GPU

# export CUDA_VISIBLE_DEVICES=1,2

# accelerate launch \
#     --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
#     --num_processes=2 \
#     --main_process_port 29501 \
#     scripts/train_omni_gen2_multi.py \
#     --config config/grpo.py:multi_omni_gen2_1gpu


# 8 GPU
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux.py --config config/grpo.py:pickscore_flux_8gpu

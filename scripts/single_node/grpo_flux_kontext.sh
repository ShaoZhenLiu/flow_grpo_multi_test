# # 1 GPU

# export CUDA_VISIBLE_DEVICES=1

# accelerate launch \
#     --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
#     --num_processes=1 \
#     --main_process_port 29501 \
#     scripts/train_flux_kontext_multi.py \
#     --config config/grpo.py:counting_flux_kontext_1gpu


# 2 GPU

export CUDA_VISIBLE_DEVICES=1,2

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes=2 \
    --main_process_port 29501 \
    scripts/train_flux_kontext_multi.py \
    --config config/grpo.py:counting_flux_kontext_1gpu


# 8 GPU
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux.py --config config/grpo.py:pickscore_flux_8gpu

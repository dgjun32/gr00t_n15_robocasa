DATASET_ROOT_PATH="/root/gr00t_n15_robocasa/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"

DATASET_LIST=(
    "/root/gr00t_n15_robocasa/datasets/4_pnp_robocasa_100"
)

export PYTHONPATH="/root/gr00t_n15_robocasa:$PYTHONPATH"

export WANDB_API_KEY='e5401e4d449b0d7fa52565da4be39dc2938c2d22'

CUDA_VISIBLE_DEVICES=1 python /root/gr00t_n15_robocasa/scripts/gr00t_finetune.py \
   --base_model_path "nvidia/GR00T-N1.5-3B" \
   --dataset-path ${DATASET_LIST[@]} \
   --num-gpus 1 \
   --max_steps 50000 \
   --output_dir /root/gr00t_n15_robocasa/checkpoints/4_pnp_robocasa_100 \
   --data_config single_panda_gripper \
   --dataloader_num_workers 8 \
   --video-backend torchvision_av \
   --save_steps 2500 \
   --gradient_accumulation_steps 4 \
   --batch_size 32
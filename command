git sparse-checkout set single_panda_gripper.CloseDoubleDoor single_panda_gripper.OpenDoubleDoor single_panda_gripper.CoffeeServeMug single_panda_gripper.CoffeeSetupMug single_panda_gripper.PnPCabToCounter single_panda_gripper.PnPCounterToCab single_panda_gripper.TurnOffSinkFaucet	single_panda_gripper.TurnOnSinkFaucet	

python gr00t/data/data_merger.py merge \
  --datasets "PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CloseDoubleDoor PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.OpenDoubleDoor PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeSetupMug PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.TurnOffSinkFaucet PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.TurnOnSinkFaucet PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.PnPCabToCounter PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.PnPCounterToCab" \
  --output_dir /root/sinjaekang/Isaac-GR00T-robocasa/0808_robocasa_multitask_sj_300 \
  --num_episodes 300 \
  --verbose


python scripts/gr00t_finetune.py \
    --dataset-path 0801_robocasa_multitask_sj_300 \
    --num-gpus 1 \
    --output-dir ~/gr00t_ckpt/test \
    --max-steps 20000 \
    --data-config single_panda_gripper \
    --batch-size 4 \
    --save-steps 1000

# Tasklist
CloseDoubleDoor 0.0 / 0.0 / 0.04
OpenDoubleDoor 0.0 / 0.0 / 
CoffeeServeMug 0.03 / 0.03 / 0.04
CoffeeSetupMug 0.0 / 0.0 / 0.0
PnPCabToCounter 0.0 / 0.0
PnPCounterToCab 0.0 / 0.0
TurnOffSinkFaucet 0.10 / 0.10
TurnOnSinkFaucet 0.03 / 0.01


conda activate gr00t_rc
CUDA_VISIBLE_DEVICES=7 python scripts/eval_policy_robocasa.py \
    --model_path checkpoint/0808/checkpoint-30000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name TurnOnSinkFaucet \
    --num_episodes 100 \
    --data_collection_path /home/sinjaekang/sinjae/Issac-GR00T-robocasa/eval_dataset/TurnOnSinkFaucet \
    --video_path /home/sinjaekang/sinjae/Issac-GR00T-robocasa/eval_video_30k_0809/TurnOnSinkFaucet \
    --max_episode_steps 1000 \
    2>&1 | tee ./logs_inference_0809_30k/TurnOnSinkFaucet_$(date +%Y%m%d_%H%M%S).log


CUDA_VISIBLE_DEVICES=7 python scripts/eval_policy_robocasa.py \
    --model_path /home/sinjaekang/sinjae/slurm_h/gr00tn15_robocasa \
    --action_horizon 16 \
    --video_backend decord \
    --dataset_path 0801_robocasa_multitask_sj_300 \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name TurnOnSinkFaucet \
    --num_episodes 100 \
    --data_collection_path /root/sinjaekang/Isaac-GR00T-robocasa/eval_dataset/test \
    --video_path /root/sinjaekang/Isaac-GR00T-robocasa/eval_video_test

  # [1, 813, 2048]
  # CloseDoubleDoor, OpenDoubleDoor, CoffeeServeMug, CoffeeSetupMug, PnPCabToCounter, PnPCounterToCab, TurnOffSinkFaucet, TurnOnSinkFaucet
    --num_episodes 100 \
    --data_collection_path /home/sinjaekang/sinjae/Issac-GR00T-robocasa/eval_dataset/TurnOnSinkFaucet \
    --video_path /home/sinjaekang/sinjae/Issac-GR00T-robocasa/eval_video_prev/TurnOnSinkFaucet \
    --max_episode_steps 2000 \
    2>&1 | tee ./logs_inference_0808_prev/TurnOnSinkFaucet_$(date +%Y%m%d_%H%M%S).log 
    
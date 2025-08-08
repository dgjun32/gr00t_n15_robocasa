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

conda activate gr00t_rc


CUDA_VISIBLE_DEVICES=3 python scripts/eval_policy_robocasa.py \
    --model_path gr00t_ckpt/0801/checkpoint-30000 \
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
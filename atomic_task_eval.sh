CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa.py \
    --model_path checkpoints/checkpoint-360000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name CloseSingleDoor \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_atomic_tasks_360k/CloseSingleDoor \
    --max_episode_steps 720


CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa.py \
    --model_path checkpoints/checkpoint-360000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name PnPSinkToCounter \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_atomic_tasks_360k/PnPSinkToCounter \
    --max_episode_steps 720


CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa.py \
    --model_path checkpoints/checkpoint-360000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name PnPCounterToMicrowave \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_atomic_tasks_360k/PnPCounterToMicrowave \
    --max_episode_steps 720


CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa.py \
    --model_path checkpoints/checkpoint-360000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name TurnOnMicrowave \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_atomic_tasks_360k/TurnOnMicrowave \
    --max_episode_steps 720


CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa.py \
    --model_path checkpoints/checkpoint-360000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name CoffeeSetupMug \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_atomic_tasks_360k/CoffeeSetupMug \
    --max_episode_steps 720


CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa.py \
    --model_path checkpoints/checkpoint-360000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name CoffeePressButton \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_atomic_tasks_360k/CoffeePressButton \
    --max_episode_steps 720


CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa.py \
    --model_path checkpoints/checkpoint-360000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name PnPCounterToSink \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_atomic_tasks_360k/PnPCounterToSink \
    --max_episode_steps 720


CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa.py \
    --model_path checkpoints/checkpoint-360000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name TurnOnSinkFaucet \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_atomic_tasks_360k/TurnOnSinkFaucet \
    --max_episode_steps 720


CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa.py \
    --model_path checkpoints/checkpoint-360000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name PnPCounterToCab \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_atomic_tasks_360k/PnPCounterToCab \
    --max_episode_steps 720


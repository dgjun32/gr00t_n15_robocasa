python scripts/eval_policy_robocasa.py \
    --model_path /root/gr00t_n15_robocasa/checkpoints/4_pnp_robocasa_100/checkpoint-10000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name PnPSinkToCab \
    --num_episodes 10 \
    --video_path /root/gr00t_n15_robocasa/eval_logs/4_pnp_robocasa_100_10000steps/PnPSinkToCab \
    --max_episode_steps 500


python scripts/eval_policy_robocasa.py \
    --model_path /root/gr00t_n15_robocasa/checkpoints/4_pnp_robocasa_100/checkpoint-10000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name PnPCounterToSink \
    --num_episodes 30 \
    --video_path /root/gr00t_n15_robocasa/eval_logs/4_pnp_robocasa_100_10000steps/PnPCounterToSink \
    --max_episode_steps 250


python scripts/eval_policy_robocasa.py \
    --model_path /root/gr00t_n15_robocasa/checkpoints/4_pnp_robocasa_100/checkpoint-10000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name PnPCounterToCab \
    --num_episodes 30 \
    --video_path /root/gr00t_n15_robocasa/eval_logs/4_pnp_robocasa_100_10000steps/PnPCounterToCab \
    --max_episode_steps 250


python scripts/eval_policy_robocasa.py \
    --model_path /root/gr00t_n15_robocasa/checkpoints/4_pnp_robocasa_100/checkpoint-10000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name PnPCabToCounter \
    --num_episodes 30 \
    --video_path /root/gr00t_n15_robocasa/eval_logs/4_pnp_robocasa_100_10000steps/PnPCabToCounter \
    --max_episode_steps 250
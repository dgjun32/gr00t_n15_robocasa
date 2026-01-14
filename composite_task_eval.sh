CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa_lh_os.py \
    --model_path /home/dongjun/Isaac-GR00T-robocasa/checkpoints/checkpoint-160000+60000_depth_1_robocasa_5_composite \
    --planner_model_name gemini-robotics-er-1.5-preview \
    --subgoal_interval 300 \
    --action_horizon 10 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name ArrangeVegetables \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_composite_tasks_gemini_os+gr00t_rc_depth_1/ArrangeVegetables \
    --max_episode_steps 3000


CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa_lh_os.py \
    --model_path /home/dongjun/Isaac-GR00T-robocasa/checkpoints/checkpoint-160000+60000_depth_1_robocasa_5_composite \
    --planner_model_name gemini-robotics-er-1.5-preview \
    --subgoal_interval 300 \
    --action_horizon 10 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name MicrowaveThawing \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_composite_tasks_gemini_os+gr00t_rc_depth_1/MicrowaveThawing \
    --max_episode_steps 3000



CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa_lh_os.py \
    --model_path /home/dongjun/Isaac-GR00T-robocasa/checkpoints/checkpoint-160000+60000_depth_1_robocasa_5_composite \
    --planner_model_name gemini-robotics-er-1.5-preview \
    --subgoal_interval 300 \
    --action_horizon 10 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name PreSoakPan \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_composite_tasks_gemini_os+gr00t_rc_depth_1/PreSoakPan \
    --max_episode_steps 3000



CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa_lh_os.py \
    --model_path /home/dongjun/Isaac-GR00T-robocasa/checkpoints/checkpoint-160000+60000_depth_1_robocasa_5_composite \
    --planner_model_name gemini-robotics-er-1.5-preview \
    --subgoal_interval 300 \
    --action_horizon 10 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name CountertopCleanup \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_composite_tasks_gemini_os+gr00t_rc_depth_1/CountertopCleanup \
    --max_episode_steps 3000
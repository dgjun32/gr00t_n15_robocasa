#!/bin/bash

NUM_EPISODES=${1:-300}

python gr00t/data/data_merger.py merge \
  --datasets "PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.PnPCabToCounter PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.PnPCounterToCab PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.PnPSinkToCounter PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.PnPCounterToSink" \
  --output_dir /root/gr00t_n15_robocasa/datasets/4_pnp_robocasa_300 \
  --num_episodes $NUM_EPISODES \
  --verbose
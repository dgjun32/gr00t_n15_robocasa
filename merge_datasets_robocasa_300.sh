#!/bin/bash

NUM_EPISODES=${1:-300}

CMD="python gr00t/data/data_merger.py merge \
  --datasets 'PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CloseDoubleDoor \
             PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CloseSingleDoor \
             PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
             PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeSetupMug \
             PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.OpenDoubleDoor \
             PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.OpenSingleDoor \
             PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.TurnOffSinkFaucet \
             PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.TurnOnSinkFaucet' \
  --output_dir /root/sinjaekang/Isaac-GR00T-robocasa/combined_dataset_300 \
  --num_episodes $NUM_EPISODES \
  --verbose"

echo "Executing command:"
echo $CMD
echo ""
eval $CMD 
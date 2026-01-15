"""
Test script for custom PnPCabToSink and PnPSinkToCab tasks
"""

import numpy as np
from robocasa.environments import REGISTERED_KITCHEN_ENVS
from robocasa.utils.env_utils import create_env

# Check if our custom tasks are registered
print("Checking if custom tasks are registered...")
print(f"PnPCabToSink registered: {'PnPCabToSink' in REGISTERED_KITCHEN_ENVS}")
print(f"PnPSinkToCab registered: {'PnPSinkToCab' in REGISTERED_KITCHEN_ENVS}")

if 'PnPCabToSink' not in REGISTERED_KITCHEN_ENVS or 'PnPSinkToCab' not in REGISTERED_KITCHEN_ENVS:
    print("\nTasks not found! Trying to import them directly...")
    # Import the module to trigger registration
    from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnPCabToSink, PnPSinkToCab
    print(f"After import - PnPCabToSink registered: {'PnPCabToSink' in REGISTERED_KITCHEN_ENVS}")
    print(f"After import - PnPSinkToCab registered: {'PnPSinkToCab' in REGISTERED_KITCHEN_ENVS}")

print("\n" + "="*60)
print("Testing PnPCabToSink Task")
print("="*60)

try:
    # Create PnPCabToSink environment
    env1 = create_env(
        env_name="PnPCabToSink",
        robots="PandaOmron",  # or your preferred robot
        controller_configs=None,  # use default
        has_renderer=False,
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=500,
    )
    
    print(f"✓ PnPCabToSink environment created successfully")
    print(f"  Action space: {env1.action_space}")
    print(f"  Observation space keys: {list(env1.observation_spec().keys())}")
    
    # Reset environment
    obs = env1.reset()
    print(f"✓ Environment reset successfully")
    
    # Get episode metadata
    ep_meta = env1.get_ep_meta()
    print(f"✓ Episode metadata:")
    print(f"    Language: {ep_meta.get('lang', 'N/A')}")
    print(f"    Layout ID: {ep_meta.get('layout_id', 'N/A')}")
    print(f"    Style ID: {ep_meta.get('style_id', 'N/A')}")
    
    # Test a few random steps
    print(f"\n  Running 10 random steps...")
    for i in range(10):
        action = env1.action_space.sample()
        obs, reward, done, info = env1.step(action)
        if done:
            print(f"    Episode finished at step {i+1}")
            break
    
    print(f"✓ PnPCabToSink task test completed successfully!\n")
    
    env1.close()
    
except Exception as e:
    print(f"✗ Error testing PnPCabToSink: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing PnPSinkToCab Task")
print("="*60)

try:
    # Create PnPSinkToCab environment
    env2 = create_env(
        env_name="PnPSinkToCab",
        robots="PandaOmron",
        controller_configs=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=500,
    )
    
    print(f"✓ PnPSinkToCab environment created successfully")
    print(f"  Action space: {env2.action_space}")
    print(f"  Observation space keys: {list(env2.observation_spec().keys())}")
    
    # Reset environment
    obs = env2.reset()
    print(f"✓ Environment reset successfully")
    
    # Get episode metadata
    ep_meta = env2.get_ep_meta()
    print(f"✓ Episode metadata:")
    print(f"    Language: {ep_meta.get('lang', 'N/A')}")
    print(f"    Layout ID: {ep_meta.get('layout_id', 'N/A')}")
    print(f"    Style ID: {ep_meta.get('style_id', 'N/A')}")
    
    # Test a few random steps
    print(f"\n  Running 10 random steps...")
    for i in range(10):
        action = env2.action_space.sample()
        obs, reward, done, info = env2.step(action)
        if done:
            print(f"    Episode finished at step {i+1}")
            break
    
    print(f"✓ PnPSinkToCab task test completed successfully!\n")
    
    env2.close()
    
except Exception as e:
    print(f"✗ Error testing PnPSinkToCab: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All tests completed!")
print("="*60)


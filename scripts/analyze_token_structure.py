# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel
from gr00t.model.transforms import build_eagle_processor
from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH

def analyze_token_structure():
    """Analyze vision token and text token positions in Eagle model."""
    
    print("=== Eagle Token Structure Analysis ===\n")
    
    # Build Eagle processor
    eagle_processor = build_eagle_processor(DEFAULT_EAGLE_PATH)
    
    # Load Eagle config to get special token info
    config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
    
    print("1. Special Token Information:")
    print(f"   Image token: {eagle_processor.image_token}")  # <IMG_CONTEXT>
    print(f"   Video token: {eagle_processor.video_token}")  # <IMG_CONTEXT>
    
    # Get tokenizer
    tokenizer = eagle_processor.tokenizer
    
    # Check image token ID
    if hasattr(config, 'image_token_index'):
        image_token_id = config.image_token_index
    else:
        # Try to get from tokenizer
        image_token_id = tokenizer.convert_tokens_to_ids(eagle_processor.image_token)
    
    print(f"   Image token ID: {image_token_id}")
    print()
    
    # Test different scenarios
    scenarios = [
        {
            "name": "Single Image + Text",
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "dummy"},
                        {"type": "text", "text": "Close the door"}
                    ]
                }
            ]
        },
        {
            "name": "Multiple Images + Text", 
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "dummy1"},
                        {"type": "image", "image": "dummy2"},
                        {"type": "text", "text": "Turn on the sink faucet"}
                    ]
                }
            ]
        },
        {
            "name": "Text Only (Empty Text)",
            "conversation": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": ""}
                    ]
                }
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}:")
        
        # Apply chat template
        formatted_text = eagle_processor.apply_chat_template(
            scenario["conversation"], 
            tokenize=False, 
            add_generation_prompt=True
        )
        print(f"   Formatted text: {repr(formatted_text)}")
        
        # Tokenize
        tokens = tokenizer(formatted_text, return_tensors="pt")
        input_ids = tokens["input_ids"][0]
        attention_mask = tokens["attention_mask"][0]
        
        print(f"   Input IDs shape: {input_ids.shape}")
        print(f"   Sequence length: {len(input_ids)}")
        
        # Find vision token positions
        vision_positions = torch.where(input_ids == image_token_id)[0]
        print(f"   Vision token positions: {vision_positions.tolist()}")
        
        # Convert tokens back to text for visualization
        token_strings = [tokenizer.decode([token_id]) for token_id in input_ids]
        
        # Show token structure
        print("   Token structure:")
        for j, (token_id, token_str, is_attended) in enumerate(zip(input_ids, token_strings, attention_mask)):
            marker = "🖼️" if token_id == image_token_id else "📝"
            attend_marker = "✓" if is_attended else "✗"
            print(f"     [{j:2d}] {marker} {attend_marker} {token_id:5d} | {repr(token_str)}")
        
        print()
    
    print("=== Token Position Analysis ===")
    print("Key findings:")
    print("1. Vision tokens (🖼️): Represented by <IMG_CONTEXT> placeholders")
    print("2. Text tokens (📝): Regular language tokens")
    print("3. Vision tokens come FIRST, then text tokens")
    print("4. Each image gets one <IMG_CONTEXT> token as placeholder")
    print("5. During forward pass, vision embeddings replace <IMG_CONTEXT> tokens")
    print()
    
    print("=== How to Extract Token Indices ===")
    print("```python")
    print("# Get vision token positions")
    print("vision_mask = (input_ids == image_token_id)")
    print("vision_indices = torch.where(vision_mask)[0]")
    print()
    print("# Get text token positions (excluding special tokens)")
    print("special_tokens = {tokenizer.bos_token_id, tokenizer.eos_token_id, ")
    print("                 tokenizer.pad_token_id, image_token_id}")
    print("text_mask = ~torch.isin(input_ids, torch.tensor(list(special_tokens)))")
    print("text_indices = torch.where(text_mask)[0]")
    print()
    print("# Split features by modality")
    print("vision_features = eagle_features[:, vision_indices, :]  # Vision embeddings")
    print("text_features = eagle_features[:, text_indices, :]      # Text embeddings")
    print("```")

def main():
    parser = argparse.ArgumentParser(description="Analyze Eagle token structure")
    args = parser.parse_args()
    
    analyze_token_structure()

if __name__ == "__main__":
    main() 
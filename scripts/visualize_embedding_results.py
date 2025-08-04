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
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_visualize_results(results_dir, output_dir=None):
    """Load saved embedding analysis results and create visualizations."""
    
    results_path = Path(results_dir)
    if output_dir is None:
        output_dir = results_path / "visualizations"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Task names (same order as in the original script)
    TASK_NAMES = [
        "CloseDoubleDoor",
        "OpenDoubleDoor", 
        "CoffeeServeMug",
        "CoffeeSetupMug",
        "PnPCabToCounter",
        "PnPCounterToCab",
        "TurnOffSinkFaucet",
        "TurnOnSinkFaucet"
    ]
    
    print(f"Loading results from {results_path}")
    print(f"Saving visualizations to {output_dir}")
    
    # 1. Load and visualize inter-task similarity
    inter_task_file = results_path / "inter_task_similarity.npy"
    if inter_task_file.exists():
        inter_task_similarity = np.load(inter_task_file)
        print(f"\nInter-task similarity matrix shape: {inter_task_similarity.shape}")
        print(f"Inter-task similarity range: {inter_task_similarity.min():.6f} - {inter_task_similarity.max():.6f}")
        
        # Calculate dynamic range for better visualization
        # Exclude diagonal (which should be 1.0)
        mask = ~np.eye(inter_task_similarity.shape[0], dtype=bool)
        off_diagonal_values = inter_task_similarity[mask]
        
        vmin = off_diagonal_values.min()
        vmax = off_diagonal_values.max()
        
        # Adjust range for better contrast - use a small margin around actual data range
        margin = (vmax - vmin) * 0.1 if vmax > vmin else 0.01
        vmin_adj = max(vmin - margin, -1.0)
        vmax_adj = min(vmax + margin, 1.0)
        
        print(f"Adjusted visualization range: {vmin_adj:.6f} - {vmax_adj:.6f}")
        
        # Create visualization with adjusted range
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(
            inter_task_similarity,
            xticklabels=TASK_NAMES,
            yticklabels=TASK_NAMES,
            annot=True,
            fmt='.4f',  # Show more decimal places
            cmap='RdYlBu_r',  # Reversed Red-Yellow-Blue for better contrast
            vmin=vmin_adj,
            vmax=vmax_adj,
            cbar_kws={'label': 'Cosine Similarity'},
            square=True
        )
        plt.title('Inter-Task Cosine Similarity Matrix\n(Higher values = more similar tasks)', 
                  fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'inter_task_similarity_adjusted.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a difference-from-maximum visualization
        plt.figure(figsize=(12, 10))
        # Show difference from maximum similarity (how much less similar each pair is)
        max_similarity = inter_task_similarity.max()
        similarity_diff = max_similarity - inter_task_similarity
        
        # Mask diagonal for this visualization
        masked_diff = np.ma.masked_where(np.eye(similarity_diff.shape[0], dtype=bool), similarity_diff)
        
        ax = sns.heatmap(
            masked_diff,
            xticklabels=TASK_NAMES,
            yticklabels=TASK_NAMES,
            annot=True,
            fmt='.6f',  # Even more decimal places for small differences
            cmap='viridis',
            cbar_kws={'label': 'Similarity Difference from Maximum'},
            square=True
        )
        plt.title('Inter-Task Similarity Differences\n(Lower values = more similar tasks)', 
                  fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'inter_task_similarity_differences.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Inter-task similarity visualizations saved")
    else:
        print(f"⚠ Inter-task similarity file not found: {inter_task_file}")
    
    # 2. Load and visualize task vs empty text similarity
    task_vs_empty_file = results_path / "task_vs_empty_text_similarity.npy"
    if task_vs_empty_file.exists():
        task_vs_empty_similarity = np.load(task_vs_empty_file)
        print(f"\nTask vs empty text similarity matrix shape: {task_vs_empty_similarity.shape}")
        print(f"Task vs empty text similarity range: {task_vs_empty_similarity.min():.6f} - {task_vs_empty_similarity.max():.6f}")
        
        # Calculate dynamic range
        vmin = task_vs_empty_similarity.min()
        vmax = task_vs_empty_similarity.max()
        margin = (vmax - vmin) * 0.1 if vmax > vmin else 0.01
        vmin_adj = max(vmin - margin, -1.0)
        vmax_adj = min(vmax + margin, 1.0)
        
        print(f"Adjusted visualization range: {vmin_adj:.6f} - {vmax_adj:.6f}")
        
        plt.figure(figsize=(14, 10))
        ax = sns.heatmap(
            task_vs_empty_similarity,
            xticklabels=[f"{name}\n(Empty Text)" for name in TASK_NAMES],
            yticklabels=[f"{name}\n(Regular Text)" for name in TASK_NAMES],
            annot=True,
            fmt='.4f',
            cmap='RdYlBu_r',
            vmin=vmin_adj,
            vmax=vmax_adj,
            cbar_kws={'label': 'Cosine Similarity'},
            square=True
        )
        plt.title('Task Embeddings vs Empty Text Embeddings Cosine Similarity\n(Higher values = language instruction has less impact)', 
                  fontsize=14, pad=20)
        plt.xlabel('Empty Text Tasks')
        plt.ylabel('Regular Text Tasks')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'task_vs_empty_text_similarity_adjusted.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create diagonal analysis (same task with/without text)
        if task_vs_empty_similarity.shape[0] == task_vs_empty_similarity.shape[1]:
            diagonal_similarities = np.diag(task_vs_empty_similarity)
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(TASK_NAMES)), diagonal_similarities, 
                          color='skyblue', edgecolor='navy', alpha=0.7)
            plt.xlabel('Tasks')
            plt.ylabel('Cosine Similarity')
            plt.title('Same Task: Regular Text vs Empty Text Similarity\n(Higher values = language instruction has less impact)')
            plt.xticks(range(len(TASK_NAMES)), TASK_NAMES, rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, diagonal_similarities)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'same_task_text_impact.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Task vs empty text similarity visualizations saved")
    else:
        print(f"⚠ Task vs empty text similarity file not found: {task_vs_empty_file}")
    
    # 2b. Visualize empty text vs empty text similarity
    empty_vs_empty_file = results_path / "empty_text_similarity.npy"
    if empty_vs_empty_file.exists():
        empty_vs_empty_similarity = np.load(empty_vs_empty_file)
        print(f"\nEmpty text vs empty text similarity matrix shape: {empty_vs_empty_similarity.shape}")
        print(f"Empty text vs empty text similarity range: {empty_vs_empty_similarity.min():.6f} - {empty_vs_empty_similarity.max():.6f}")
        
        # Calculate dynamic range
        mask = ~np.eye(empty_vs_empty_similarity.shape[0], dtype=bool)
        off_diagonal_values = empty_vs_empty_similarity[mask]
        
        vmin = off_diagonal_values.min()
        vmax = off_diagonal_values.max()
        margin = (vmax - vmin) * 0.1 if vmax > vmin else 0.01
        vmin_adj = max(vmin - margin, -1.0)
        vmax_adj = min(vmax + margin, 1.0)
        
        print(f"Adjusted visualization range: {vmin_adj:.6f} - {vmax_adj:.6f}")
        
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(
            empty_vs_empty_similarity,
            xticklabels=[f"{name}\n(Empty Text)" for name in TASK_NAMES],
            yticklabels=[f"{name}\n(Empty Text)" for name in TASK_NAMES],
            annot=True,
            fmt='.4f',
            cmap='RdYlBu_r',
            vmin=vmin_adj,
            vmax=vmax_adj,
            cbar_kws={'label': 'Cosine Similarity'},
            square=True
        )
        plt.title('Empty Text Tasks Inter-Similarity Matrix\n(How similar tasks are when language instruction is removed)', 
                  fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'empty_text_similarity_adjusted.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create difference visualization for empty text
        plt.figure(figsize=(12, 10))
        max_similarity = empty_vs_empty_similarity.max()
        similarity_diff = max_similarity - empty_vs_empty_similarity
        masked_diff = np.ma.masked_where(np.eye(similarity_diff.shape[0], dtype=bool), similarity_diff)
        
        ax = sns.heatmap(
            masked_diff,
            xticklabels=[f"{name}\n(Empty Text)" for name in TASK_NAMES],
            yticklabels=[f"{name}\n(Empty Text)" for name in TASK_NAMES],
            annot=True,
            fmt='.6f',
            cmap='viridis',
            cbar_kws={'label': 'Similarity Difference from Maximum'},
            square=True
        )
        plt.title('Empty Text Tasks Similarity Differences\n(Lower values = more similar without language cues)', 
                  fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'empty_text_similarity_differences.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Empty text vs empty text similarity visualizations saved")
    else:
        print(f"⚠ Empty text similarity file not found: {empty_vs_empty_file}")
    
    # 3. Load and visualize intra-task similarity statistics
    stats_file = results_path / "summary_statistics.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            summary_stats = json.load(f)
        
        print(f"\nLoaded summary statistics")
        
        # Visualize intra-task similarities
        if 'intra_task_similarities' in summary_stats:
            intra_stats = summary_stats['intra_task_similarities']
            
            if intra_stats:
                tasks = list(intra_stats.keys())
                means = [intra_stats[task]['mean'] for task in tasks]
                stds = [intra_stats[task]['std'] for task in tasks]
                mins = [intra_stats[task]['min'] for task in tasks]
                maxs = [intra_stats[task]['max'] for task in tasks]
                
                # Create bar plot with error bars
                plt.figure(figsize=(14, 8))
                x = np.arange(len(tasks))
                
                bars = plt.bar(x, means, yerr=stds, capsize=5, 
                              color='lightcoral', edgecolor='darkred', alpha=0.7)
                plt.xlabel('Tasks')
                plt.ylabel('Intra-Task Cosine Similarity')
                plt.title('Intra-Task Similarity Statistics\n(Higher values = more consistent embeddings within same task)')
                plt.xticks(x, tasks, rotation=45, ha='right')
                plt.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for i, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.001, 
                            f'{mean_val:.4f}', ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(output_dir / 'intra_task_similarity_stats.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create detailed comparison table
                plt.figure(figsize=(16, 10))
                stats_data = []
                for task in tasks:
                    stats_data.append([
                        task,
                        f"{intra_stats[task]['mean']:.6f}",
                        f"{intra_stats[task]['std']:.6f}",
                        f"{intra_stats[task]['min']:.6f}",
                        f"{intra_stats[task]['max']:.6f}"
                    ])
                
                table = plt.table(cellText=stats_data,
                                colLabels=['Task', 'Mean', 'Std', 'Min', 'Max'],
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0, 1, 1])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 2)
                
                # Style the table
                for i in range(len(tasks) + 1):
                    for j in range(5):
                        if i == 0:  # Header
                            table[(i, j)].set_facecolor('#40466e')
                            table[(i, j)].set_text_props(weight='bold', color='white')
                        else:
                            table[(i, j)].set_facecolor('#f1f1f2')
                
                plt.axis('off')
                plt.title('Intra-Task Similarity Statistics Summary', fontsize=16, pad=20)
                plt.tight_layout()
                plt.savefig(output_dir / 'intra_task_similarity_table.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✓ Intra-task similarity visualizations saved")
            else:
                print("⚠ No intra-task similarity data found in summary statistics")
        
        # Print detailed statistics
        print(f"\n=== Detailed Statistics ===")
        if 'inter_task_similarity' in summary_stats and summary_stats['inter_task_similarity']['mean'] is not None:
            inter_stats = summary_stats['inter_task_similarity']
            print(f"\nInter-task similarity statistics:")
            print(f"  Mean: {inter_stats['mean']:.6f}")
            print(f"  Std:  {inter_stats['std']:.6f}")
            print(f"  Min:  {inter_stats['min']:.6f}")
            print(f"  Max:  {inter_stats['max']:.6f}")
        
        if 'intra_task_similarities' in summary_stats and summary_stats['intra_task_similarities']:
            print(f"\nIntra-task similarity statistics:")
            for task_name, stats in summary_stats['intra_task_similarities'].items():
                print(f"  {task_name}:")
                print(f"    Mean: {stats['mean']:.6f}")
                print(f"    Std:  {stats['std']:.6f}")
                print(f"    Min:  {stats['min']:.6f}")
                print(f"    Max:  {stats['max']:.6f}")
    else:
        print(f"⚠ Summary statistics file not found: {stats_file}")
    
    print(f"\n✅ All visualizations saved to {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - inter_task_similarity_adjusted.png")
    print(f"  - inter_task_similarity_differences.png")  
    print(f"  - task_vs_empty_text_similarity_adjusted.png")
    print(f"  - same_task_text_impact.png")
    print(f"  - empty_text_similarity_adjusted.png")
    print(f"  - empty_text_similarity_differences.png")
    print(f"  - intra_task_similarity_stats.png")
    print(f"  - intra_task_similarity_table.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize saved embedding analysis results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./embedding_analysis_results",
        help="Directory containing the saved .npy and .json result files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default=None,
        help="Directory to save visualizations (default: results_dir/visualizations)"
    )
    
    args = parser.parse_args()
    
    load_and_visualize_results(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main() 
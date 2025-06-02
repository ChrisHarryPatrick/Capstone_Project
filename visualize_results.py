import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

def load_fl_results(filename="fl_results_transformed.json"):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

import matplotlib.pyplot as plt
import numpy as np

def create_combined_accuracy_loss_chart(data):
    rounds = list(range(1, len(data["accuracy_history"]) + 1))
    accuracy = data["accuracy_history"]
    loss = data["loss_history"]
    aggregation_rounds = data["aggregation_rounds"]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot accuracy on primary axis
    color = 'tab:blue'
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy', color=color)
    accuracy_line = ax1.plot(rounds, accuracy, color=color, label='Accuracy', marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)  # Accuracy range 0-1
    
    # Create secondary axis for loss
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color)
    loss_line = ax2.plot(rounds, loss, color=color, label='Loss', marker='x')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add aggregation markers
    for r in aggregation_rounds:
        if r <= len(rounds):
            ax1.axvline(x=r, color='gray', linestyle=':', alpha=0.5)
    
    # Combine legends from both axes
    lines = accuracy_line + loss_line
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # Add title and grid
    plt.title('Model Accuracy and Loss per Round')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('combined_accuracy_loss_chart.png')
    plt.close()
    print("✅ Saved combined accuracy/loss chart to combined_accuracy_loss_chart.png")



def plot_client_participation_with_accuracy(data):
    plt.figure(figsize=(14, 7))
    
    rounds = np.arange(1, len(data["accuracy_history"]) + 1)
    num_clients = max([int(k) for k in data["client_participation"].keys()]) + 1
    client_colors = plt.cm.tab10.colors
    marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
    
    for cid in range(num_clients):
        client_id = str(cid)
        participation = data["client_participation"].get(client_id, [False]*len(rounds))
        participation = participation[:len(rounds)] + [False]*(len(rounds)-len(participation))
        
        accuracies = []
        for r in range(len(rounds)):
            if participation[r]:
                accuracies.append(data["accuracy_history"][r])
            else:
                accuracies.append(np.nan)
        
        plt.plot(rounds, accuracies,
                 color=client_colors[cid % len(client_colors)],
                 linestyle='-',
                 marker=marker_styles[cid % len(marker_styles)],
                 markersize=8,
                 label=f'Client {cid}',
                 alpha=0.8,
                 markeredgecolor='white',
                 markeredgewidth=1)
        
        part_rounds = rounds[np.array(participation)]
        part_acc = np.array(accuracies)[np.array(participation)]
        plt.scatter(part_rounds, part_acc,
                    color=client_colors[cid % len(client_colors)],
                    s=100,
                    edgecolors='black',
                    zorder=3)

        skip_rounds = rounds[~np.array(participation)]
        for r in skip_rounds:
            plt.text(r, -0.05, '✖', 
                     ha='center', 
                     va='center',
                     color=client_colors[cid % len(client_colors)],
                     fontsize=10,
                     alpha=0.5)
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Client Participation and Accuracy Trends', fontsize=14, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.05)
    plt.xlim(0.5, len(rounds)+0.5)

    plt.tight_layout()
    plt.savefig('client_participation_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved improved client participation/accuracy chart with all clients shown.")



def main():
    # Create output directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    
    # Load the FL results
    try:
        data = load_fl_results()
    except FileNotFoundError:
        print("❌ Error: fl_results.json not found. Run the FL experiment first.")
        return
    
    # Generate charts
    create_combined_accuracy_loss_chart(data)
    plot_client_participation_with_accuracy(data)
    
    print("\n🎉 Visualization complete! Check the current directory for:")
    print("- accuracy_loss_chart.png")
    print("- client_participation_distribution.png")

if __name__ == "__main__":
    main()
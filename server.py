import flwr as fl
import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime
import matplotlib.pyplot as plt
import os
from model import EnhancedCNNModel, get_model_params, set_model_params
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client_types = {0: "fast", 1: "medium", 2: "slow", 3: "fast", 4: "medium"}

class ServerMetrics:
    def __init__(self):
        self.history = {
            "accuracy": [],
            "loss": [],
            "round_times": [],
            "participations": defaultdict(list),
            "client_metrics": [],
            "aggregation_freq": [],
            "config_history": [],
            "best_accuracy": 0.0,
            "best_model_round": 0,
            "best_model_params": None,
            "client_participation": defaultdict(list),
            "class_distributions": defaultdict(list),
            "aggregation_rounds": [],
            "round_class_distributions" : defaultdict(dict),
            "timestamps": []
        }
    
    def update_best_model(self, accuracy, parameters, round_num):
        if accuracy > self.history["best_accuracy"]:
            self.history["best_accuracy"] = accuracy
            self.history["best_model_round"] = round_num
            self.history["best_model_params"] = parameters
            print(f"🔥 New best model at Round {round_num} - Accuracy: {accuracy:.2%}")
    
    def save(self, filename="fl_results.json"):
        save_data = {
            "accuracy_history": self.history["accuracy"],
            "loss_history": self.history["loss"],
            "round_times": self.history["round_times"],
            "participations": dict(self.history["participations"]),
            "config_history": self.history["config_history"],
            "best_accuracy": self.history["best_accuracy"],
            "best_model_round": self.history["best_model_round"],
            "client_participation": dict(self.history["client_participation"]),
            "class_distributions": dict(self.history["class_distributions"]),
            "aggregation_rounds": self.history["aggregation_rounds"],
            "timestamps": self.history["timestamps"]
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        if self.history["best_model_params"] is not None:
            torch.save(
                self.history["best_model_params"],
                f"best_model_r{self.history['best_model_round']}_acc{self.history['best_accuracy']:.4f}.pt"
            )
        
        self.generate_visualizations()
        print(f"✅ Metrics saved to {filename}")
        print(f"✅ Best model (Round {self.history['best_model_round']}) saved to PT file")
    
    def generate_visualizations(self):
        os.makedirs("visualizations", exist_ok=True)
        rounds = list(range(1, len(self.history["accuracy"]) + 1))
        accuracy = self.history["accuracy"]
        loss = self.history["loss"]
        aggregation_rounds = self.history["aggregation_rounds"]
        
        # Chart 1: Accuracy and Loss
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(rounds, accuracy, 'b-', label='Accuracy')
        for r in aggregation_rounds:
            if r <= len(accuracy):
                plt.axvline(x=r, color='r', linestyle='--', alpha=0.3)
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy per Round')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(rounds, loss, 'r-', label='Loss')
        for r in aggregation_rounds:
            if r <= len(loss):
                plt.axvline(x=r, color='b', linestyle='--', alpha=0.3)
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Model Loss per Round')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('visualizations/accuracy_loss_chart.png')
        plt.close()
        
        # Chart 2: Client Participation Heatmap
        participation_data = []
        max_rounds = len(rounds)
        
        for cid in range(5):
            participation = []
            for round_num in rounds:
                if round_num-1 < len(self.history["client_participation"].get(cid, [])):
                    participation.append(1 if self.history["client_participation"][cid][round_num-1] else 0)
                else:
                    participation.append(0)
            participation_data.append(participation)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(participation_data, cmap='Blues', aspect='auto')
        plt.colorbar(label='Participation (1=yes, 0=no)')
        plt.xlabel('Round')
        plt.ylabel('Client ID')
        plt.title('Client Participation Heatmap')
        plt.yticks(range(5), [f'Client {i}' for i in range(5)])
        plt.savefig('visualizations/client_participation_heatmap.png')
        plt.close()
        
        # Chart 3: Class Distribution for Client 0
        if 0 in self.history["class_distributions"]:
            class_data = self.history["class_distributions"][0]
            class_counts = []
            
            for class_str in class_data:
                if class_str:
                    classes = list(map(int, class_str.split(',')))
                    counts = {cls: 0 for cls in range(10)}
                    counts.update(Counter(classes))
                    class_counts.append([counts[cls] for cls in range(10)])
            
            if class_counts:
                plt.figure(figsize=(12, 6))
                plt.stackplot(range(1, len(class_counts)+1), 
                            np.array(class_counts).T,
                            labels=[f'Class {i}' for i in range(10)])
                plt.xlabel('Round')
                plt.ylabel('Number of Samples')
                plt.title('Class Distribution for Client 0 Over Rounds')
                plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
                plt.grid(True)
                plt.savefig('visualizations/class_distribution_client0.png', bbox_inches='tight')
                plt.close()

        print("✅ Generated visualization charts in 'visualizations' directory")

metrics = ServerMetrics()

def weighted_average(metrics_list: List[Tuple[int, Dict]]) -> Dict:
    accuracies = []
    losses = []
    total_examples = 0
    
    # First pass: Record participation and class distributions
    for num_examples, m in metrics_list:
        if m:
            if "cid" in m:
                cid = m["cid"]
                client_type = client_types[cid]
                metrics.history["participations"][client_type].append(1)
                
                # Track class distributions for participating clients
                if "classes" in m and "server_round" in m:
                    round_num = m["server_round"]
                    metrics.history["round_class_distributions"][round_num][cid] = m["classes"]
            
            # For metrics aggregation
            if "accuracy" in m:
                accuracies.append(m["accuracy"] * num_examples)
            if "loss" in m:
                losses.append(m["loss"] * num_examples)
            total_examples += num_examples
    
    # Second pass: Ensure all clients are accounted for in each round
    if metrics_list and "server_round" in metrics_list[0][1]:
        current_round = metrics_list[0][1]["server_round"]
        
        # Initialize participation for all clients this round
        for cid in range(5):  # Assuming 5 clients (0-4)
            if cid not in metrics.history["round_class_distributions"][current_round]:
                metrics.history["round_class_distributions"][current_round][cid] = None  # Mark as skipped
    
    # Calculate weighted averages
    aggregated = {}
    if accuracies and total_examples > 0:
        aggregated["accuracy"] = sum(accuracies) / total_examples
    if losses and total_examples > 0:
        aggregated["loss"] = sum(losses) / total_examples
    
    return aggregated

class WaitForClientsStrategy(fl.server.strategy.FedAvg):
    def __init__(self, min_clients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_clients = min_clients
        self.ready = False
        self.aggregation_frequency = 2
        self.last_accuracy = 0.0
        self.patience = 2
        self.current_parameters = None

    def initialize_parameters(self, client_manager):
        while len(client_manager.all()) < self.min_clients:
            time.sleep(1)
        self.ready = True
        initial_params = super().initialize_parameters(client_manager)
        self.current_parameters = initial_params
        return initial_params

    def should_aggregate(self, server_round: int) -> bool:
        if server_round <= 4:
            return True
        return server_round % self.aggregation_frequency == 0

    def update_aggregation_frequency(self, current_accuracy):
        if current_accuracy <= self.last_accuracy:
            self.aggregation_frequency = 1
            print("⚠️ Accuracy stagnated — reverting to aggregation every round.")
        else:
            self.aggregation_frequency = 2
        self.last_accuracy = current_accuracy

    def aggregate_fit(self, server_round, results, failures):
        metrics.history["timestamps"].append(datetime.now().isoformat())
        
        # Initialize round tracking
        metrics.history["round_class_distributions"][server_round] = {}
        
        # Process results
        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Update best model tracking if this was an aggregation round
        if self.should_aggregate(server_round):
            metrics.history["aggregation_rounds"].append(server_round)
            self.current_parameters = aggregated_params
            
            # Update aggregation frequency based on accuracy
            if server_round-1 < len(metrics.history["accuracy"]):
                self.update_aggregation_frequency(metrics.history["accuracy"][server_round-1])
        else:
            print(f"🔄 Skipping aggregation at Round {server_round}")
        
        return aggregated_params, aggregated_metrics

    def configure_fit(self, server_round, parameters, client_manager):
        if not self.ready:
            return []
        
        # Initialize participation tracking for this round
        for cid in range(5):  # Assuming 5 clients
            if server_round-1 >= len(metrics.history["client_participation"].get(cid, [])):
                metrics.history["client_participation"][cid].append(False)
        
        self.current_parameters = parameters
        return super().configure_fit(server_round, parameters, client_manager)

def get_evaluate_fn():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        net = EnhancedCNNModel().to(device)
        set_model_params(net, parameters)
        
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        net.eval()
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.argmax(outputs, 1) == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        metrics.history["accuracy"].append(accuracy)
        metrics.history["loss"].append(loss)
        
        metrics.update_best_model(accuracy, parameters, server_round)
        
        print(f"\n[Round {server_round}] Server Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")
        print(f"Best Accuracy So Far: {metrics.history['best_accuracy']:.2%} (Round {metrics.history['best_model_round']})")
        
        torch.cuda.empty_cache()
        return loss, {"accuracy": accuracy}
    
    return evaluate

def fit_config(server_round: int):
    lr = max(0.0005, 0.001 * (0.95 ** server_round))
    epochs = 2
    
    config = {
        "server_round": server_round,
        "lr": lr,
        "epochs": epochs,
        "batch_size": 32,
    }
    metrics.history["config_history"].append(config)
    return config

def main():
    model = EnhancedCNNModel().cpu()
    initial_params = fl.common.ndarrays_to_parameters(get_model_params(model))
    
    min_clients = 5
    
    strategy = WaitForClientsStrategy(
        min_clients=min_clients,
        initial_parameters=initial_params,
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=4,
        min_evaluate_clients=2,
        min_available_clients=min_clients,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print("🚀 Starting Federated Learning Server...")
    start_time = time.time()
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=15),
        strategy=strategy,
    )
    
    metrics.save()
    
    total_time = time.time() - start_time
    print(f"\n🏁 Training Complete!")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print(f"🏆 Best Accuracy: {metrics.history['best_accuracy']:.2%} (Round {metrics.history['best_model_round']})")

if __name__ == "__main__":
    main()
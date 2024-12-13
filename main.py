import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(thread_execution_times):
    single_thread_time = thread_execution_times[0][1]
    metrics = []
    for num_threads, exec_time in thread_execution_times:
        speedup = single_thread_time / exec_time if exec_time > 0 else 0
        efficiency = speedup / num_threads if num_threads > 0 else 0
        scalability = speedup / np.log2(num_threads) if num_threads > 1 else 1
        parallel_overhead = (num_threads * exec_time - single_thread_time) / single_thread_time
        metrics.append({
            "num_threads": num_threads,
            "execution_time": exec_time,
            "speedup": speedup,
            "efficiency": efficiency,
            "scalability": scalability,
            "parallel_overhead": parallel_overhead
        })
    return metrics

def run_benchmark(benchmark_function, dataset_configs, thread_counts):
    results = {}
    for n_samples, n_features in dataset_configs:
        print(f"Processing dataset with {n_samples} samples and {n_features} features...")
        thread_execution_times = benchmark_function(n_samples, n_features, thread_counts)
        results[(n_samples, n_features)] = calculate_metrics(thread_execution_times)
    return results

def plot_results(results, title):
    metrics = ["speedup", "efficiency", "scalability", "parallel_overhead"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{title} - Performance Metrics", fontsize=16)
    
    for (n_samples, n_features), data in results.items():
        label = f'{n_samples}x{n_features}'
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            x = [d["num_threads"] for d in data]
            y = [d[metric] for d in data]
            ax.plot(x, y, marker='o', label=label)
            ax.set_title(metric.capitalize())
            ax.set_xlabel("Number of Threads")
            ax.set_ylabel(metric.capitalize())
            ax.legend(title="Dataset Size (samples x features)")
            ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_metrics.png")
    plt.close()

# Modify the main function to use the new plotting function
if _name_ == "_main_":
    args = parse_args()

    for algo in args.algorithms:
        if algo == "pca":
            results = run_benchmark(run_pca_benchmark, args.dataset_configs, args.thread_counts)
            plot_results(results, "PCA")
        elif algo == "rf":
            results = run_benchmark(run_random_forest_benchmark, args.dataset_configs, args.thread_counts)
            plot_results(results, "Random Forest")
        elif algo == "nn":
            results = run_benchmark(run_neural_network_benchmark, args.dataset_configs, args.thread_counts)
            plot_results(results, "Neural Network")
        elif algo == "knn":
            results = run_benchmark(run_knn_benchmark, args.dataset_configs, args.thread_counts)
            plot_results(results, "K-Nearest Neighbors")

    print("Benchmarking complete. Results saved as PNG files.")

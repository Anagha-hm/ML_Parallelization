import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import multiprocessing

class ParallelDecisionTreeBenchmark:
    @staticmethod
    def generate_dataset(n_samples, n_features, random_state=42):
        """
        Generate a synthetic classification dataset with robust feature handling.
        
        Args:
            n_samples (int): Number of samples to generate
            n_features (int): Total number of features
            random_state (int): Random seed for reproducibility
        
        Returns:
            tuple: X (features), y (labels)
        """
        # Adjust informative features to be less than total features
        n_informative = max(2, min(n_features // 2, 10))
        n_redundant = max(0, min(n_features - n_informative, 2))
        n_repeated = max(0, min(n_features - n_informative - n_redundant, 1))
        
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features, 
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            random_state=random_state
        )
        return X, y

    @staticmethod
    def train_decision_tree_sequential(X_train, y_train, X_test, y_test, max_depth=None):
        """
        Train a decision tree sequentially.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_test (np.ndarray): Testing features
            y_test (np.ndarray): Testing labels
            max_depth (int, optional): Maximum depth of the tree
        
        Returns:
            tuple: Accuracy, training time
        """
        start_time = time.time()
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X_train, y_train)
        
        # Predict and calculate accuracy
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        training_time = time.time() - start_time
        return accuracy, training_time

    @staticmethod
    def train_decision_tree_parallel(X_train, y_train, X_test, y_test, n_jobs=-1, max_depth=None):
        """
        Train decision trees in parallel using joblib.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_test (np.ndarray): Testing features
            y_test (np.ndarray): Testing labels
            n_jobs (int): Number of parallel jobs (-1 uses all available cores)
            max_depth (int, optional): Maximum depth of the trees
        
        Returns:
            tuple: Accuracy, training time
        """
        start_time = time.time()
        
        # Create multiple estimators
        def create_estimator():
            return DecisionTreeClassifier(max_depth=max_depth)
        
        # Parallel training
        clfs = Parallel(n_jobs=n_jobs)(
            delayed(create_estimator().fit)(X_train, y_train) 
            for _ in range(multiprocessing.cpu_count())
        )
        
        # Predict using the first classifier (or implement ensemble prediction)
        y_pred = clfs[0].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        training_time = time.time() - start_time
        return accuracy, training_time

    def run_benchmark(self, sample_sizes=[1000, 5000, 10000, 50000], 
                      feature_sizes=[10, 20, 50], 
                      thread_counts=None):
        """
        Run comprehensive benchmarking for decision tree training.
        
        Args:
            sample_sizes (list): List of dataset sizes to test
            feature_sizes (list): List of feature sizes to test
            thread_counts (list): List of thread counts to test
        
        Returns:
            dict: Benchmark results
        """
        if thread_counts is None:
            thread_counts = [1, 2, 4, max(1, multiprocessing.cpu_count())]
        
        results = {
            'sequential': {},
            'parallel': {}
        }
        
        for n_samples in sample_sizes:
            for n_features in feature_sizes:
                # Generate dataset
                X, y = self.generate_dataset(n_samples, n_features)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Sequential benchmark
                seq_accuracy, seq_time = self.train_decision_tree_sequential(
                    X_train, y_train, X_test, y_test
                )
                
                results['sequential'][(n_samples, n_features)] = {
                    'accuracy': seq_accuracy,
                    'time': seq_time
                }
                
                # Parallel benchmarks
                parallel_results = {}
                for n_jobs in thread_counts:
                    par_accuracy, par_time = self.train_decision_tree_parallel(
                        X_train, y_train, X_test, y_test, n_jobs=n_jobs
                    )
                    
                    speedup = seq_time / par_time if par_time > 0 else 1
                    
                    parallel_results[n_jobs] = {
                        'accuracy': par_accuracy,
                        'time': par_time,
                        'speedup': speedup
                    }
                
                results['parallel'][(n_samples, n_features)] = parallel_results
        
        return results

    def plot_results(self, results):
        """
        Visualize benchmark results.
        
        Args:
            results (dict): Benchmark results from run_benchmark
        """
        plt.figure(figsize=(15, 10))
        
        # Speedup plot
        plt.subplot(2, 2, 1)
        for (n_samples, n_features), parallel_data in results['parallel'].items():
            thread_counts = list(parallel_data.keys())
            speedups = [data['speedup'] for data in parallel_data.values()]
            plt.plot(thread_counts, speedups, 
                     label=f'Samples: {n_samples}, Features: {n_features}')
        
        plt.title('Parallel Speedup')
        plt.xlabel('Number of Threads')
        plt.ylabel('Speedup')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Training Time plot
        plt.subplot(2, 2, 2)
        for (n_samples, n_features), parallel_data in results['parallel'].items():
            thread_counts = list(parallel_data.keys())
            times = [data['time'] for data in parallel_data.values()]
            plt.plot(thread_counts, times, 
                     label=f'Samples: {n_samples}, Features: {n_features}')
        
        plt.title('Training Time')
        plt.xlabel('Number of Threads')
        plt.ylabel('Time (seconds)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()

# Example usage and demonstration
if __name__ == "__main__":
    benchmark = ParallelDecisionTreeBenchmark()
    results = benchmark.run_benchmark()
    benchmark.plot_results(results)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, load_wine, load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class ModelComparison:
    def __init__(self, cv_folds=5, random_state=42, test_size=0.2):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.results = {}
        
        # Define models to compare
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=random_state),
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'K-Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state)
        }
        
        # Define scoring metrics
        self.scoring_metrics = {
            'accuracy': 'accuracy',
            'precision': 'precision_macro',
            'recall': 'recall_macro', 
            'f1': 'f1_macro',
            'roc_auc': 'roc_auc_ovr_weighted'
        }
    
    def load_sample_dataset(self, dataset_name='wine'):
        """Load sample datasets for demonstration"""
        if dataset_name == 'wine':
            data = load_wine()
            return data.data, data.target, data.feature_names, data.target_names
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            return data.data, data.target, data.feature_names, data.target_names
        elif dataset_name == 'synthetic':
            X, y = make_classification(
                n_samples=1000, n_features=20, n_informative=15,
                n_redundant=5, n_classes=3, random_state=self.random_state
            )
            feature_names = [f'feature_{i}' for i in range(20)]
            target_names = ['Class_0', 'Class_1', 'Class_2']
            return X, y, feature_names, target_names
        else:
            raise ValueError("Supported datasets: 'wine', 'breast_cancer', 'synthetic'")
    
    def prepare_data(self, X, y, scale_features=True):
        """Prepare data for model comparison"""
        if scale_features:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        return X_scaled, y
    
    def perform_cross_validation(self, X, y):
        """Perform cross-validation for all models"""
        print("Performing cross-validation comparison...")
        print("=" * 60)
        
        # Initialize results storage
        cv_results = {}
        detailed_results = {}
        
        # Set up stratified k-fold (better for imbalanced datasets)
        cv_strategy = StratifiedKFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            # Perform cross-validation with multiple metrics
            cv_scores = cross_validate(
                model, X, y,
                cv=cv_strategy,
                scoring=self.scoring_metrics,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Store results
            detailed_results[model_name] = cv_scores
            cv_results[model_name] = {
                'accuracy_mean': cv_scores['test_accuracy'].mean(),
                'accuracy_std': cv_scores['test_accuracy'].std(),
                'precision_mean': cv_scores['test_precision'].mean(),
                'precision_std': cv_scores['test_precision'].std(),
                'recall_mean': cv_scores['test_recall'].mean(),
                'recall_std': cv_scores['test_recall'].std(),
                'f1_mean': cv_scores['test_f1'].mean(),
                'f1_std': cv_scores['test_f1'].std(),
                'roc_auc_mean': cv_scores['test_roc_auc'].mean(),
                'roc_auc_std': cv_scores['test_roc_auc'].std(),
                'fit_time_mean': cv_scores['fit_time'].mean(),
                'score_time_mean': cv_scores['score_time'].mean()
            }
        
        self.results = cv_results
        self.detailed_results = detailed_results
        return cv_results
    
    def create_results_dataframe(self):
        """Convert results to pandas DataFrame for easier analysis"""
        results_list = []
        
        for model_name, metrics in self.results.items():
            results_list.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}",
                'Precision': f"{metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}",
                'Recall': f"{metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}",
                'F1-Score': f"{metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}",
                'Fit Time (s)': f"{metrics['fit_time_mean']:.4f}",
                'Score Time (s)': f"{metrics['score_time_mean']:.4f}"
            })
        
        return pd.DataFrame(results_list)
    
    def plot_model_comparison(self):
        """Create comprehensive visualization of model performance"""
        if not self.results:
            print("No results to plot. Run cross-validation first.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison - Cross Validation Results', fontsize=16)
        
        # Metrics to plot
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            if idx < 5:  # Plot first 5 metrics
                row = idx // 3
                col = idx % 3
                ax = axes[row, col]
                
                # Extract data for plotting
                models = list(self.results.keys())
                means = [self.results[model][f'{metric}_mean'] for model in models]
                stds = [self.results[model][f'{metric}_std'] for model in models]
                
                # Create bar plot with error bars
                bars = ax.bar(range(len(models)), means, yerr=stds, capsize=5, alpha=0.7)
                ax.set_xlabel('Models')
                ax.set_ylabel(label)
                ax.set_title(f'{label} Comparison')
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels([model.replace(' ', '\n') for model in models], rotation=0)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std,
                           f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Training time comparison
        ax = axes[1, 2]
        models = list(self.results.keys())
        fit_times = [self.results[model]['fit_time_mean'] for model in models]
        
        bars = ax.bar(range(len(models)), fit_times, alpha=0.7, color='orange')
        ax.set_xlabel('Models')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Training Time Comparison')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([model.replace(' ', '\n') for model in models], rotation=0)
        ax.grid(True, alpha=0.3)
        
        for i, (bar, time) in enumerate(zip(bars, fit_times)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{time:.3f}s', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def plot_score_distribution(self):
        """Plot distribution of cross-validation scores"""
        if not self.detailed_results:
            print("No detailed results to plot. Run cross-validation first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy distribution
        ax1 = axes[0]
        accuracy_data = []
        model_labels = []
        
        for model_name, scores in self.detailed_results.items():
            accuracy_data.extend(scores['test_accuracy'])
            model_labels.extend([model_name] * len(scores['test_accuracy']))
        
        accuracy_df = pd.DataFrame({
            'Model': model_labels,
            'Accuracy': accuracy_data
        })
        
        sns.boxplot(data=accuracy_df, x='Model', y='Accuracy', ax=ax1)
        ax1.set_title('Cross-Validation Accuracy Distribution')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # F1-Score distribution
        ax2 = axes[1]
        f1_data = []
        model_labels = []
        
        for model_name, scores in self.detailed_results.items():
            f1_data.extend(scores['test_f1'])
            model_labels.extend([model_name] * len(scores['test_f1']))
        
        f1_df = pd.DataFrame({
            'Model': model_labels,
            'F1-Score': f1_data
        })
        
        sns.boxplot(data=f1_df, x='Model', y='F1-Score', ax=ax2)
        ax2.set_title('Cross-Validation F1-Score Distribution')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def statistical_significance_test(self):
        """Perform statistical tests to compare model performances"""
        from scipy.stats import friedmanchisquare, wilcoxon
        
        if not self.detailed_results:
            print("No detailed results for statistical testing.")
            return
        
        print("\nStatistical Significance Testing")
        print("=" * 40)
        
        # Friedman test (non-parametric test for multiple related samples)
        accuracy_scores = [scores['test_accuracy'] for scores in self.detailed_results.values()]
        stat, p_value = friedmanchisquare(*accuracy_scores)
        
        print(f"Friedman Test for Accuracy:")
        print(f"Test Statistic: {stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("Result: Significant difference between models (p < 0.05)")
        else:
            print("Result: No significant difference between models (p >= 0.05)")
        
        # Pairwise comparisons (Wilcoxon signed-rank test)
        print(f"\nPairwise Comparisons (Wilcoxon Test):")
        print("-" * 40)
        
        model_names = list(self.detailed_results.keys())
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                scores1 = self.detailed_results[model1]['test_accuracy']
                scores2 = self.detailed_results[model2]['test_accuracy']
                
                try:
                    stat, p_val = wilcoxon(scores1, scores2, alternative='two-sided')
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"{model1} vs {model2}: p-value = {p_val:.6f} {significance}")
                except:
                    print(f"{model1} vs {model2}: Test not applicable (identical scores)")
    
    def get_best_models(self, metric='accuracy', top_n=3):
        """Get top performing models based on specified metric"""
        if not self.results:
            print("No results available. Run cross-validation first.")
            return
        
        # Sort models by specified metric
        sorted_models = sorted(
            self.results.items(), 
            key=lambda x: x[1][f'{metric}_mean'], 
            reverse=True
        )
        
        print(f"\nTop {top_n} Models by {metric.upper()}:")
        print("=" * 40)
        
        for i, (model_name, metrics) in enumerate(sorted_models[:top_n]):
            mean_score = metrics[f'{metric}_mean']
            std_score = metrics[f'{metric}_std']
            print(f"{i+1}. {model_name}: {mean_score:.4f} ± {std_score:.4f}")
        
        return sorted_models[:top_n]
    
    def run_complete_comparison(self, X=None, y=None, dataset_name='wine', scale_features=True):
        """Run complete model comparison pipeline"""
        print("Machine Learning Model Comparison with Cross-Validation")
        print("=" * 60)
        
        # Load data if not provided
        if X is None or y is None:
            X, y, feature_names, target_names = self.load_sample_dataset(dataset_name)
            print(f"Dataset: {dataset_name}")
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        print(f"Cross-validation folds: {self.cv_folds}")
        
        # Prepare data
        X_processed, y_processed = self.prepare_data(X, y, scale_features)
        
        # Perform cross-validation
        cv_results = self.perform_cross_validation(X_processed, y_processed)
        
        # Create results DataFrame
        results_df = self.create_results_dataframe()
        print(f"\n{results_df.to_string(index=False)}")
        
        # Get best models
        self.get_best_models('accuracy', top_n=3)
        self.get_best_models('f1', top_n=3)
        
        # Create visualizations
        self.plot_model_comparison()
        self.plot_score_distribution()
        
        # Statistical significance testing
        self.statistical_significance_test()
        
        return results_df

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize comparison
    comparator = ModelComparison(cv_folds=5, random_state=42)
    
    # Run comparison on wine dataset
    print("Running comparison on Wine dataset...")
    results_wine = comparator.run_complete_comparison(dataset_name='wine')
    
    print("\n" + "="*80)
    print("Running comparison on Breast Cancer dataset...")
    
    # Run on different dataset
    comparator2 = ModelComparison(cv_folds=10, random_state=42)  # More CV folds
    results_cancer = comparator2.run_complete_comparison(dataset_name='breast_cancer')
    
    # Example with custom data
    print("\n" + "="*80)
    print("Running comparison on Synthetic dataset...")
    
    # Create custom dataset
    from sklearn.datasets import make_classification
    X_custom, y_custom = make_classification(
        n_samples=2000, n_features=25, n_informative=20,
        n_redundant=5, n_classes=4, random_state=42
    )
    
    comparator3 = ModelComparison(cv_folds=5, random_state=42)
    results_custom = comparator3.run_complete_comparison(X=X_custom, y=y_custom)
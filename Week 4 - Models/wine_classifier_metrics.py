import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StrategyifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score, log_loss,
    brier_score_loss, auc, average_precision_score
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

class WineClassifierEvaluator:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_wine_data(self):
        """Load and prepare wine datasets"""
        try:
            # Try to load from UCI repository URLs
            red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
            
            red_wine = pd.read_csv(red_url, sep=';')
            white_wine = pd.read_csv(white_url, sep=';')
            
            # Add wine type column
            red_wine['wine_type'] = 'red'
            white_wine['wine_type'] = 'white'
            
            # Combine datasets
            wine_data = pd.concat([red_wine, white_wine], ignore_index=True)
            
        except:
            # Generate synthetic wine data if URLs are not accessible
            print("Creating synthetic wine data for demonstration...")
            np.random.seed(42)
            n_samples = 2000
            
            # Generate synthetic features similar to wine quality dataset
            data = {
                'fixed acidity': np.random.normal(8.3, 1.7, n_samples),
                'volatile acidity': np.random.normal(0.5, 0.18, n_samples),
                'citric acid': np.random.normal(0.3, 0.15, n_samples),
                'residual sugar': np.random.lognormal(1.5, 1.2, n_samples),
                'chlorides': np.random.normal(0.08, 0.05, n_samples),
                'free sulfur dioxide': np.random.normal(35, 17, n_samples),
                'total sulfur dioxide': np.random.normal(140, 42, n_samples),
                'density': np.random.normal(0.997, 0.003, n_samples),
                'pH': np.random.normal(3.2, 0.15, n_samples),
                'sulphates': np.random.normal(0.65, 0.17, n_samples),
                'alcohol': np.random.normal(10.4, 1.1, n_samples),
                'wine_type': np.random.choice(['red', 'white'], n_samples, p=[0.3, 0.7])
            }
            
            wine_data = pd.DataFrame(data)
            # Create quality based on features (synthetic relationship)
            quality_score = (
                wine_data['alcohol'] * 0.3 +
                wine_data['sulphates'] * 2 +
                -wine_data['volatile acidity'] * 5 +
                wine_data['citric acid'] * 2 +
                np.random.normal(0, 0.5, n_samples)
            )
            wine_data['quality'] = np.clip(np.round(quality_score).astype(int), 3, 9)
        
        return wine_data
    
    def create_classification_tasks(self, wine_data):
        """Create different classification tasks"""
        tasks = {}
        
        # Task 1: Wine Type Classification (Red vs White)
        if 'wine_type' in wine_data.columns:
            X_type = wine_data.drop(['quality', 'wine_type'], axis=1)
            y_type = wine_data['wine_type']
            tasks['Wine Type'] = (X_type, y_type)
        
        # Task 2: Quality Classification (Low/Medium/High)
        X_quality = wine_data.drop(['quality'] + (['wine_type'] if 'wine_type' in wine_data.columns else []), axis=1)
        y_quality_binned = pd.cut(wine_data['quality'], 
                                bins=[0, 5, 6, 10], 
                                labels=['Low', 'Medium', 'High'])
        tasks['Quality (3-class)'] = (X_quality, y_quality_binned)
        
        # Task 3: High Quality Binary Classification (Quality >= 7)
        y_high_quality = (wine_data['quality'] >= 7).astype(int)
        tasks['High Quality'] = (X_quality, y_high_quality)
        
        return tasks
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba=None, task_type='multiclass'):
        """Calculate comprehensive classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Balanced Accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Handle different averaging strategies based on task type
        if task_type == 'binary':
            metrics['Precision'] = precision_score(y_true, y_pred)
            metrics['Recall'] = recall_score(y_true, y_pred)
            metrics['F1-Score'] = f1_score(y_true, y_pred)
            if y_pred_proba is not None:
                metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                metrics['PR-AUC'] = average_precision_score(y_true, y_pred_proba[:, 1])
        else:
            metrics['Precision (Macro)'] = precision_score(y_true, y_pred, average='macro')
            metrics['Precision (Weighted)'] = precision_score(y_true, y_pred, average='weighted')
            metrics['Recall (Macro)'] = recall_score(y_true, y_pred, average='macro')
            metrics['Recall (Weighted)'] = recall_score(y_true, y_pred, average='weighted')
            metrics['F1-Score (Macro)'] = f1_score(y_true, y_pred, average='macro')
            metrics['F1-Score (Weighted)'] = f1_score(y_true, y_pred, average='weighted')
            if y_pred_proba is not None:
                try:
                    metrics['ROC-AUC (Macro)'] = roc_auc_score(y_true, y_pred_proba, average='macro', multi_class='ovr')
                    metrics['ROC-AUC (Weighted)'] = roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovr')
                except:
                    pass
        
        # Additional metrics
        metrics['Matthews Correlation'] = matthews_corrcoef(y_true, y_pred)
        metrics['Cohen Kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Probabilistic metrics (if probabilities available)
        if y_pred_proba is not None:
            try:
                metrics['Log Loss'] = log_loss(y_true, y_pred_proba)
                if task_type == 'binary':
                    metrics['Brier Score'] = brier_score_loss(y_true, y_pred_proba[:, 1])
            except:
                pass
        
        return metrics
    
    def plot_confusion_matrices(self, task_name, y_true, predictions):
        """Plot confusion matrices for all models"""
        n_models = len(predictions)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Confusion Matrices - {task_name}', fontsize=16)
        axes = axes.flatten()
        
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{model_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide empty subplot
        if len(predictions) < len(axes):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, task_name, y_true, probabilities, task_type='multiclass'):
        """Plot ROC curves for binary classification tasks"""
        if task_type != 'binary':
            return
            
        plt.figure(figsize=(10, 8))
        
        for model_name, y_pred_proba in probabilities.items():
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {task_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_precision_recall_curves(self, task_name, y_true, probabilities, task_type='multiclass'):
        """Plot Precision-Recall curves for binary classification tasks"""
        if task_type != 'binary':
            return
            
        plt.figure(figsize=(10, 8))
        
        for model_name, y_pred_proba in probabilities.items():
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - {task_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def evaluate_all_models(self, tasks):
        """Evaluate all models on all tasks"""
        for task_name, (X, y) in tasks.items():
            print(f"\n{'='*60}")
            print(f"Evaluating Task: {task_name}")
            print(f"{'='*60}")
            
            # Determine task type
            task_type = 'binary' if len(np.unique(y)) == 2 else 'multiclass'
            
            # Encode labels if necessary
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            task_results = {}
            predictions = {}
            probabilities = {}
            
            print(f"Dataset shape: {X.shape}")
            print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
            print("\nModel Performance:")
            print("-" * 80)
            
            for model_name, model in self.models.items():
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self.calculate_comprehensive_metrics(
                    y_test, y_pred, y_pred_proba, task_type
                )
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                metrics['CV Mean'] = cv_scores.mean()
                metrics['CV Std'] = cv_scores.std()
                
                task_results[model_name] = metrics
                predictions[model_name] = y_pred
                if y_pred_proba is not None:
                    probabilities[model_name] = y_pred_proba
                
                # Print key metrics
                print(f"{model_name:18} | "
                      f"Acc: {metrics['Accuracy']:.3f} | "
                      f"F1: {metrics.get('F1-Score', metrics.get('F1-Score (Weighted)', 0)):.3f} | "
                      f"CV: {metrics['CV Mean']:.3f}±{metrics['CV Std']:.3f}")
            
            self.results[task_name] = task_results
            
            # Generate visualizations
            print("\nGenerating visualizations...")
            self.plot_confusion_matrices(task_name, y_test, predictions)
            self.plot_roc_curves(task_name, y_test, probabilities, task_type)
            self.plot_precision_recall_curves(task_name, y_test, probabilities, task_type)
    
    def create_results_summary(self):
        """Create a comprehensive results summary"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE RESULTS SUMMARY")
        print(f"{'='*80}")
        
        for task_name, task_results in self.results.items():
            print(f"\n{task_name.upper()} CLASSIFICATION")
            print("-" * 60)
            
            # Create DataFrame for better formatting
            results_df = pd.DataFrame(task_results).T
            
            # Sort by accuracy
            results_df = results_df.sort_values('Accuracy', ascending=False)
            
            # Display key metrics
            key_metrics = ['Accuracy', 'Balanced Accuracy', 'CV Mean', 'Matthews Correlation']
            if 'F1-Score' in results_df.columns:
                key_metrics.append('F1-Score')
            elif 'F1-Score (Weighted)' in results_df.columns:
                key_metrics.append('F1-Score (Weighted)')
            
            print(results_df[key_metrics].round(4).to_string())
            
            # Best model
            best_model = results_df.index[0]
            print(f"\nBest Model: {best_model} (Accuracy: {results_df.loc[best_model, 'Accuracy']:.4f})")
    
    def run_complete_evaluation(self):
        """Run the complete evaluation pipeline"""
        print("Wine Classification Model Evaluation")
        print("="*50)
        
        # Load data
        print("Loading wine dataset...")
        wine_data = self.load_wine_data()
        print(f"Dataset loaded with shape: {wine_data.shape}")
        
        # Create classification tasks
        print("Creating classification tasks...")
        tasks = self.create_classification_tasks(wine_data)
        print(f"Created {len(tasks)} classification tasks: {list(tasks.keys())}")
        
        # Evaluate models
        self.evaluate_all_models(tasks)
        
        # Summary
        self.create_results_summary()

# Run the evaluation
if __name__ == "__main__":
    evaluator = WineClassifierEvaluator()
    evaluator.run_complete_evaluation()
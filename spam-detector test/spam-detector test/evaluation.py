import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


class SpamClassifierEvaluator:
   
    #Comprehensive evaluator for spam classification models
    

    def __init__(self):
        #Initialize evaluator
        self.results = {}
        self.training_history = {}

    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Comprehensive model evaluation

        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            model_name (str): Name of the model

        Returns:
            dict: Evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

        # Add AUC score if probabilities are available
        try:
            if hasattr(y_pred, 'predict_proba'):
                y_proba = y_pred.predict_proba(y_true)[:, 1]
                metrics['auc_score'] = roc_auc_score(y_true, y_proba)
        except:
            pass

        self.results[model_name] = metrics
        return metrics

    def evaluate_rl_training(self, agent):
        """
        Evaluate RL agent training progress

        Args:
            agent: Trained RL agent

        Returns:
            dict: Training evaluation metrics
        """
        history = agent.training_history

        training_metrics = {
            'final_accuracy': np.mean(history['episode_accuracy'][-10:]),
            'max_accuracy': max(history['episode_accuracy']),
            'accuracy_improvement': (
                np.mean(history['episode_accuracy'][-10:]) - 
                np.mean(history['episode_accuracy'][:10])
            ),
            'final_reward': np.mean(history['episode_rewards'][-10:]),
            'max_reward': max(history['episode_rewards']),
            'convergence_episode': self._find_convergence_point(history['episode_accuracy']),
            'exploration_decay': history['epsilon_values'][0] - history['epsilon_values'][-1]
        }

        self.training_history = training_metrics
        return training_metrics

    def _find_convergence_point(self, accuracies, window=10, threshold=0.01):
        #Find approximate convergence point in training
        if len(accuracies) < window * 2:
            return len(accuracies)

        for i in range(window, len(accuracies) - window):
            recent_avg = np.mean(accuracies[i:i+window])
            past_avg = np.mean(accuracies[i-window:i])

            if abs(recent_avg - past_avg) < threshold:
                return i

        return len(accuracies)

    def plot_training_progress(self, agent, save_path=None):
        """
        Plot RL training progress

        Args:
            agent: Trained RL agent
            save_path (str): Path to save plot
        """
        history = agent.training_history

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy over episodes
        axes[0, 0].plot(history['episode_accuracy'], 'b-', alpha=0.7)
        axes[0, 0].plot(self._smooth(history['episode_accuracy']), 'r-', linewidth=2)
        axes[0, 0].set_title('Training Accuracy')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)

        # Rewards over episodes
        axes[0, 1].plot(history['episode_rewards'], 'g-', alpha=0.7)
        axes[0, 1].plot(self._smooth(history['episode_rewards']), 'r-', linewidth=2)
        axes[0, 1].set_title('Training Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Reward')
        axes[0, 1].grid(True, alpha=0.3)

        # Epsilon decay
        axes[1, 0].plot(history['epsilon_values'], 'purple', linewidth=2)
        axes[1, 0].set_title('Epsilon Decay (Exploration)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True, alpha=0.3)

        # Learning progress (accuracy vs reward)
        axes[1, 1].scatter(history['episode_rewards'], history['episode_accuracy'], 
                          alpha=0.6, c=range(len(history['episode_rewards'])), cmap='viridis')
        axes[1, 1].set_title('Learning Progress (Accuracy vs Reward)')
        axes[1, 1].set_xlabel('Episode Reward')
        axes[1, 1].set_ylabel('Episode Accuracy')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_model_comparison(self, save_path=None):
        """
        Plot comparison of different models

        Args:
            save_path (str): Path to save plot
        """
        if not self.results:
            print("No model results to plot")
            return

        # Prepare data for plotting
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']

        data = []
        for model in models:
            for metric in metrics:
                if metric in self.results[model]:
                    data.append({
                        'Model': model,
                        'Metric': metric.title(),
                        'Score': self.results[model][metric]
                    })

        df = pd.DataFrame(data)

        # Create comparison plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Model', y='Score', hue='Metric')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_confusion_matrices(self, save_path=None):
        """
        Plot confusion matrices for all evaluated models

        Args:
            save_path (str): Path to save plot
        """
        models_with_cm = [m for m in self.results.keys() 
                         if 'confusion_matrix' in self.results[m]]

        if not models_with_cm:
            print("No confusion matrices to plot")
            return

        n_models = len(models_with_cm)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for i, model in enumerate(models_with_cm):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            cm = self.results[model]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model}\nConfusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        # Hide empty subplots
        for i in range(n_models, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def _smooth(self, data, window=5):
        #Apply moving average smoothing
        if len(data) < window:
            return data

        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            smoothed.append(np.mean(data[start:end]))

        return smoothed

    def generate_report(self, save_path=None):
        """
        Generate comprehensive evaluation report

        Args:
            save_path (str): Path to save report

        Returns:
            str: Evaluation report
        """
        report = []
        report.append("=" * 60)
        report.append("SPAM EMAIL CLASSIFICATION - EVALUATION REPORT")
        report.append("=" * 60)

        # Model comparison
        if self.results:
            report.append("\nMODEL PERFORMANCE COMPARISON")
            report.append("-" * 40)

            # Sort models by accuracy
            sorted_models = sorted(self.results.items(), 
                                 key=lambda x: x[1]['accuracy'], reverse=True)

            for i, (model, metrics) in enumerate(sorted_models, 1):
                report.append(f"{i}. {model}:")
                report.append(f"   Accuracy:  {metrics['accuracy']:.4f}")
                report.append(f"   Precision: {metrics['precision']:.4f}")
                report.append(f"   Recall:    {metrics['recall']:.4f}")
                report.append(f"   F1-Score:  {metrics['f1_score']:.4f}")
                report.append("")

        # RL training analysis
        if self.training_history:
            report.append("\nRL TRAINING ANALYSIS")
            report.append("-" * 40)
            th = self.training_history
            report.append(f"Final Accuracy: {th['final_accuracy']:.4f}")
            report.append(f"Maximum Accuracy: {th['max_accuracy']:.4f}")
            report.append(f"Accuracy Improvement: {th['accuracy_improvement']:.4f}")
            report.append(f"Final Reward: {th['final_reward']:.2f}")
            report.append(f"Maximum Reward: {th['max_reward']:.2f}")
            report.append(f"Convergence Episode: {th['convergence_episode']}")
            report.append(f"Exploration Decay: {th['exploration_decay']:.4f}")

        report.append("\n" + "=" * 60)

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)

        return report_text


def quick_evaluate_rl_agent(agent, X_test, y_test):
    """
    Quick evaluation of RL agent on test data

    Args:
        agent: Trained RL agent
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Evaluation metrics
    """
    predictions = []

    # Use agent to predict on test data
    for i in range(len(X_test)):
        # Set epsilon to 0 for testing (no exploration)
        original_epsilon = agent.epsilon
        agent.epsilon = 0

        action = agent.act(X_test[i])
        predictions.append(action)

        # Restore epsilon
        agent.epsilon = original_epsilon

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, average='weighted'),
        'recall': recall_score(y_test, predictions, average='weighted'),
        'f1_score': f1_score(y_test, predictions, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, predictions)
    }

    return metrics, predictions


if __name__ == "__main__":
    print("Spam Classifier Evaluation Tools")
    print("=" * 40)
    print("Available evaluation functions:")
    print("- SpamClassifierEvaluator: Comprehensive evaluation class")
    print("- quick_evaluate_rl_agent: Quick RL agent evaluation")
    print("- Visualization tools for training progress and comparisons")

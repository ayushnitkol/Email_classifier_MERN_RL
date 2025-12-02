import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import deque
import re
import warnings
warnings.filterwarnings('ignore')


class SpamClassificationEnvironment:
    """
    Reinforcement Learning Environment for Spam Email Classification

    This environment simulates the spam classification task where:
    - State: Email features (TF-IDF vectors)
    - Actions: 0 (classify as ham), 1 (classify as spam)
    - Rewards: +1 for correct classification, -1 for incorrect
    """

    def __init__(self, X, y):
        """
        Initialize the environment

        Args:
            X (numpy.ndarray): Feature matrix (TF-IDF vectors)
            y (numpy.ndarray): True labels (0=ham, 1=spam)
        """
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.current_index = 0
        self.episode_rewards = []
        self.episode_actions = []
        self.correct_predictions = 0
        self.total_predictions = 0

    def reset(self):
        #Reset environment to initial state
        self.current_index = 0
        self.episode_rewards = []
        self.episode_actions = []
        self.correct_predictions = 0
        self.total_predictions = 0

        # Shuffle data for each episode
        indices = np.random.permutation(self.n_samples)
        self.X = self.X[indices]
        self.y = self.y[indices]

        return self.X[self.current_index]

    def step(self, action):
        """
        Execute action and return next state, reward, done flag, and info

        Args:
            action (int): Action taken by agent (0=ham, 1=spam)

        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Get true label for current email
        true_label = self.y[self.current_index]

        # Calculate reward
        if action == true_label:
            reward = 1.0  # Correct classification
            self.correct_predictions += 1
        else:
            reward = -1.0  # Incorrect classification

        self.total_predictions += 1
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)

        # Move to next email
        self.current_index += 1

        # Check if episode is done
        done = self.current_index >= self.n_samples

        # Get next state
        next_state = None if done else self.X[self.current_index]

        # Additional info
        info = {
            'accuracy': self.correct_predictions / self.total_predictions,
            'total_reward': sum(self.episode_rewards)
        }

        return next_state, reward, done, info


class QLearningSpamAgent:
    """
    Q-Learning Agent for Spam Email Classification

    This agent uses Q-Learning algorithm to learn optimal policies
    for classifying emails as spam or ham.
    """

    def __init__(self, action_size=2, learning_rate=0.1, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01, gamma=0.95):
        """
        Initialize Q-Learning agent

        Args:
            action_size (int): Number of possible actions
            learning_rate (float): Learning rate for Q-updates
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Decay rate for epsilon
            epsilon_min (float): Minimum epsilon value
            gamma (float): Discount factor for future rewards
        """
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        # Q-table as dictionary
        self.q_table = {}

        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_accuracy': [],
            'epsilon_values': []
        }

    def get_state_key(self, state):
        #Convert continuous state to discrete key for Q-table
        # Discretize continuous features by taking top features and binning them
        top_features = np.argsort(state)[-5:]
        state_key = tuple(np.digitize(state[top_features], bins=np.linspace(0, 1, 6)))
        return state_key

    def get_q_value(self, state, action):
        #Get Q-value for state-action pair
        state_key = self.get_state_key(state)
        if (state_key, action) not in self.q_table:
            self.q_table[(state_key, action)] = 0.0
        return self.q_table[(state_key, action)]

    def act(self, state):
        #Choose action using epsilon-greedy policy
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values = [self.get_q_value(state, action) for action in range(self.action_size)]
        return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done):
        #Update Q-value using Q-learning update rule
        current_q = self.get_q_value(state, action)

        if done or next_state is None:
            target_q = reward
        else:
            next_q_values = [self.get_q_value(next_state, a) for a in range(self.action_size)]
            target_q = reward + self.gamma * max(next_q_values)

        # Q-learning update
        state_key = self.get_state_key(state)
        self.q_table[(state_key, action)] = current_q + self.learning_rate * (target_q - current_q)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def preprocess_text(text):
    #Preprocess email text
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text


def create_sample_dataset():
    #Create sample spam/ham email dataset
    spam_emails = [
        "Congratulations! You've won $1000! Click here to claim now!",
        "FREE MONEY! Win big cash prizes! Act now!",
        "URGENT: Your account will be closed. Click to verify immediately!",
        "Amazing offer! Buy one get one free! Limited time only!",
        "You are a winner! Claim your lottery prize now!",
        "Hot singles in your area! Click to meet them!",
        "Make money fast! Work from home opportunity!",
        "Nigerian prince needs your help! Big reward!",
        "Credit repair! Bad credit? No problem! Call now!"
    ]

    ham_emails = [
        "Hi John, let's meet for lunch tomorrow at 12pm",
        "Thank you for your purchase. Your order has been confirmed",
        "Meeting scheduled for Monday at 3pm in conference room A",
        "Your monthly statement is now available online",
        "Happy birthday! Hope you have a wonderful day",
        "The project deadline has been extended to Friday",
        "Your flight confirmation for Flight AA123 tomorrow",
        "New product features are now available in your dashboard",
        "Team dinner on Friday at 7pm. Please confirm attendance",
        "Your subscription renewal is due in 5 days"
    ]

    emails = spam_emails + ham_emails
    labels = ['spam'] * len(spam_emails) + ['ham'] * len(ham_emails)

    data = pd.DataFrame({'email': emails, 'label': labels})
    data['label_binary'] = data['label'].map({'spam': 1, 'ham': 0})
    data['email_processed'] = data['email'].apply(preprocess_text)

    return data


def train_rl_agent(agent, env, episodes=100):
    """Train the RL agent"""
    print(f"Training RL agent for {episodes} episodes...")

    training_scores = []
    training_accuracies = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        training_scores.append(total_reward)
        training_accuracies.append(info['accuracy'])
        agent.training_history['episode_rewards'].append(total_reward)
        agent.training_history['episode_accuracy'].append(info['accuracy'])
        agent.training_history['epsilon_values'].append(agent.epsilon)

        if episode % 10 == 0:
            avg_score = np.mean(training_scores[-10:])
            avg_accuracy = np.mean(training_accuracies[-10:])
            print(f"Episode {episode:3d} | Score: {avg_score:6.2f} | Accuracy: {avg_accuracy:.3f}")

    return training_scores, training_accuracies


def compare_algorithms(X, y, rl_accuracy):
    """Compare RL with traditional ML algorithms"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    algorithms = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42)
    }

    results = {'RL Agent (Q-Learning)': rl_accuracy}

    for name, algorithm in algorithms.items():
        algorithm.fit(X_train, y_train)
        y_pred = algorithm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

    return results


if __name__ == "__main__":
    # Set random seeds
    random.seed(42)
    np.random.seed(42)

    print("Spam Email Classification with Reinforcement Learning")
    print("=" * 60)

    # Create dataset
    print("Creating sample dataset...")
    data = create_sample_dataset()

    # Feature extraction
    print("Extracting features using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(data['email_processed']).toarray()
    y = data['label_binary'].values

    # Create environment and agent
    print("🏗️  Setting up RL environment and agent...")
    env = SpamClassificationEnvironment(X, y)
    agent = QLearningSpamAgent(
        action_size=2,
        learning_rate=0.1,
        epsilon=1.0,
        epsilon_decay=0.98,
        epsilon_min=0.05,
        gamma=0.95
    )

    # Train agent
    print("Training RL agent...")
    scores, accuracies = train_rl_agent(agent, env, episodes=50)

    # Compare with traditional methods
    print("Comparing with traditional ML methods...")
    rl_accuracy = np.mean(accuracies[-10:])
    comparison = compare_algorithms(X, y, rl_accuracy)

    print("\nResults Summary:")
    print("-" * 40)
    for method, accuracy in sorted(comparison.items(), key=lambda x: x[1], reverse=True):
        print(f"{method}: {accuracy:.3f}")

    print(f"\nTraining completed!")
    print(f" Q-table size: {len(agent.q_table)} state-action pairs")

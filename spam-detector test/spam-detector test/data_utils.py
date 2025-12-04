import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter


class EmailDataProcessor:
    """
    Email Data Processor for spam classification
    Handles loading, preprocessing, and feature extraction
    """

    def __init__(self, max_features=1000, use_tfidf=True):
        """
        Initialize data processor

        Args:
            max_features (int): Maximum number of features to extract
            use_tfidf (bool): Whether to use TF-IDF or Count Vectorizer
        """
        self.max_features = max_features
        self.use_tfidf = use_tfidf
        self.vectorizer = None
        self.spam_keywords = []
        self.ham_keywords = []

    def preprocess_text(self, text):
        """
        Comprehensive text preprocessing

        Args:
            text (str): Raw email text

        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def extract_keywords(self, texts, labels, top_n=20):
        """
        Extract top keywords for spam and ham emails

        Args:
            texts (list): List of email texts
            labels (list): List of labels (0=ham, 1=spam)
            top_n (int): Number of top keywords to extract
        """
        spam_texts = [texts[i] for i in range(len(texts)) if labels[i] == 1]
        ham_texts = [texts[i] for i in range(len(texts)) if labels[i] == 0]

        # Get word frequencies
        spam_words = ' '.join(spam_texts).split()
        ham_words = ' '.join(ham_texts).split()

        spam_counter = Counter(spam_words)
        ham_counter = Counter(ham_words)

        self.spam_keywords = [word for word, count in spam_counter.most_common(top_n)]
        self.ham_keywords = [word for word, count in ham_counter.most_common(top_n)]

    def create_features(self, texts):
        """
        Create feature vectors from email texts

        Args:
            texts (list): List of preprocessed email texts

        Returns:
            numpy.ndarray: Feature matrix
        """
        if self.vectorizer is None:
            if self.use_tfidf:
                self.vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            else:
                self.vectorizer = CountVectorizer(
                    max_features=self.max_features,
                    stop_words='english',
                    ngram_range=(1, 2)
                )

            # Fit vectorizer
            X = self.vectorizer.fit_transform(texts).toarray()
        else:
            # Transform using existing vectorizer
            X = self.vectorizer.transform(texts).toarray()

        return X

    def load_enron_dataset(self, file_path):
        """
        Load Enron spam dataset

        Args:
            file_path (str): Path to Enron dataset CSV

        Returns:
            tuple: (features, labels, raw_data)
        """
        try:
            data = pd.read_csv(file_path)

            # Assume columns are: 'Subject', 'Message', 'Spam/Ham'
            if 'Message' in data.columns and 'Spam/Ham' in data.columns:
                texts = data['Message'].fillna('').astype(str)
                labels = data['Spam/Ham'].map({'ham': 0, 'spam': 1})
            else:
                raise ValueError("Dataset must have 'Message' and 'Spam/Ham' columns")

            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]

            # Create features
            X = self.create_features(processed_texts)
            y = labels.values

            return X, y, data

        except Exception as e:
            print(f"Error loading Enron dataset: {e}")
            return None, None, None

    def create_synthetic_dataset(self, n_samples=1000):
        """
        Create synthetic spam/ham dataset for testing

        Args:
            n_samples (int): Number of samples to generate

        Returns:
            tuple: (features, labels, texts)
        """
        # Spam patterns
        spam_templates = [
            "congratulations you have won ${} click here to claim",
            "free money win big cash prizes act now",
            "urgent your account will be closed verify immediately",
            "amazing offer buy one get one free limited time",
            "you are a winner claim your prize now",
            "hot singles in your area click to meet",
            "make money fast work from home opportunity",
            "cheap medications online no prescription needed",
            "get rich quick system guaranteed results",
            "credit repair bad credit no problem call now"
        ]

        # Ham patterns
        ham_templates = [
            "hi {} let's meet for lunch tomorrow",
            "thank you for your purchase order confirmed",
            "meeting scheduled for monday at {} in conference room",
            "your monthly statement is now available online",
            "happy birthday hope you have wonderful day",
            "project deadline has been extended to friday",
            "your flight confirmation for flight {} tomorrow",
            "new product features available in dashboard",
            "team dinner on friday please confirm attendance",
            "your subscription renewal is due in {} days"
        ]

        texts = []
        labels = []

        # Generate spam emails
        for _ in range(n_samples // 2):
            template = np.random.choice(spam_templates)
            # Add variation
            if '{}' in template:
                if '${}' in template:
                    text = template.format(np.random.randint(100, 10000))
                else:
                    text = template.format(np.random.choice(['john', 'mary', 'bob', 'alice']))
            else:
                text = template
            texts.append(text)
            labels.append(1)  # spam

        # Generate ham emails
        for _ in range(n_samples // 2):
            template = np.random.choice(ham_templates)
            if '{}' in template:
                if 'flight {}' in template:
                    text = template.format(f"AA{np.random.randint(100, 999)}")
                elif 'at {}' in template:
                    text = template.format(f"{np.random.randint(9, 17)}pm")
                elif '{} days' in template:
                    text = template.format(np.random.randint(1, 30))
                else:
                    text = template.format(np.random.choice(['john', 'mary', 'bob', 'alice']))
            else:
                text = template
            texts.append(text)
            labels.append(0)  # ham

        # Shuffle data
        combined = list(zip(texts, labels))
        np.random.shuffle(combined)
        texts, labels = zip(*combined)

        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Create features
        X = self.create_features(processed_texts)
        y = np.array(labels)

        return X, y, list(texts)


def load_spam_dataset(dataset_name='synthetic', **kwargs):
    """
    Load spam dataset by name

    Args:
        dataset_name (str): Name of dataset ('synthetic', 'enron', 'spambase')
        **kwargs: Additional arguments for dataset loading

    Returns:
        tuple: (X, y, processor)
    """
    processor = EmailDataProcessor(**kwargs)

    if dataset_name == 'synthetic':
        n_samples = kwargs.get('n_samples', 1000)
        X, y, texts = processor.create_synthetic_dataset(n_samples)
        return X, y, processor

    elif dataset_name == 'enron':
        file_path = kwargs.get('file_path', 'enron_spam_data.csv')
        X, y, data = processor.load_enron_dataset(file_path)
        return X, y, processor

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    print("Email Data Processing Utilities")
    print("=" * 40)

    # Test synthetic dataset creation
    print("Creating synthetic dataset...")
    processor = EmailDataProcessor(max_features=100)
    X, y, texts = processor.create_synthetic_dataset(n_samples=100)

    print(f"Dataset shape: {X.shape}")
    print(f"Labels distribution: {Counter(y)}")
    print(f"Sample texts: {texts[:3]}")

    # Extract keywords
    print("\nExtracting keywords...")
    processor.extract_keywords(texts, y, top_n=10)
    print(f"Top spam keywords: {processor.spam_keywords[:5]}")
    print(f"Top ham keywords: {processor.ham_keywords[:5]}")

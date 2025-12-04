import pandas as pd
from data_utils import EmailDataProcessor

dataset_path= "data/SMSSpamcollection"

df = pd.read.csv(dataset_path)

texts = df["text"].tolist()
labels = df["label"].tolist()

processor = EmailDataProcessor()

X= processor.create_features(texts)
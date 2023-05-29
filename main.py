import pandas as pd
import numpy as np

import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

review: str = "I am so happy"

df = pd.read_csv("dataset/Reviews.csv")
# print(df.head(5))
# print(df.columns)


def vader_model(text:str) -> dict:
    """Perform sentiment anlysis with vedas
    Args:
        text (str): The text chart to be used for the sentiment analysis

    Returns:
        dict: _description_
    """
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)


def main():
    model1 = vader_model(review)
    print(model1)

if __name__ == '__main__':
    main()
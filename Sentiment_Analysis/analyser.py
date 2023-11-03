import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SentimentAnalyser:

    def __init__(self):
        self.__analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(self, text):

        scores = self.__analyzer.polarity_scores(text)

        sentiment = 1 if scores['pos'] > 0 else 0

        return sentiment


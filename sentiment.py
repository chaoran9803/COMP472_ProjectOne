from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        #load sentiment analysis model, this modle is trained on twitter data
        self.model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    def analyze(self, text):
        result = self.model(text)[0]
        
        #we need the .upper to match the statment in the main.py
        label = result['label'].upper()
        score = result['score']
        return label, score
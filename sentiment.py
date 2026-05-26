from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    def analyze(self, text):
        result = self.model(text)[0]
        
        #we need the .upper to match the statment in the main.py
        label = result['label'].upper()
        score = result['score']
        if label == 'NEGATIVE' and score > 0.7:
            score = 0.95
        return label, score
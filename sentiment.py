from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.model = pipeline("sentiment-analysis")

    def analyze(self, text):
        result = self.model(text)[0]
        return result['label'], result['score']
    
    def check_escalation(label,score):
        if label == 'NEGATIVE' and score > 0.9:
            return "We recommend contacting a human advisor."
        return None
from embedding import EmbeddingEngine
from sentiment import SentimentAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from search import find_best_answer
import numpy as np

def check_escalation(label,score):
        if label == 'NEGATIVE' and score > 0.9:
            return "We recommend contacting a human advisor."
        return None

def main():
    print("Welcome to the student support chatbot!")
    print("Type 'quit' to exit.\n")

    engine = EmbeddingEngine()
    sentiment = SentimentAnalyzer()

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        # Sentiment analysis
        label, score = sentiment.analyze(user_input)
        print(f"Sentiment: {label} ({score:.2f})")

        # Escalation
        escalation = check_escalation(label, score)
        if escalation:
            print("Recommended escalation:", escalation)

        # Semantic search
        answer, sim = find_best_answer(engine, user_input)
        print("Answer:", answer)
        print()

if __name__ == "__main__":
    main()
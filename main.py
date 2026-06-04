from embedding import EmbeddingEngine
from sentiment import SentimentAnalyzer
from search import find_best_answer

# Simple escalation logic based on sentiment analysis results
def check_escalation(label,score):
        if label == 'NEGATIVE' and score > 0.9:
            return "Please contact a human advisor."
        return None

def main():

    engine = EmbeddingEngine()
    sentiment = SentimentAnalyzer()

    print("\n Welcome to the student support AI chatbot!")
    print("Type 'quit' to exit.\n")
    
    # The loop will continue until the user types 'quit'
    while True:
        user_input = input("Your Question: ")
        if user_input.lower() == "quit":
            print("Bye Bye butterfly!")
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
        print(f"Answer:{answer}\n")

if __name__ == "__main__":
    main()
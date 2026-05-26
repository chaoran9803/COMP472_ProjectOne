import pandas as pd
from sentence_transformers import SentenceTransformer

class EmbeddingEngine:
    def __init__(self, csv_path = "knowledge_base.csv"):

        #load csv
        df = pd.read_csv(csv_path)
        self.questions = df['question'].astype(str).tolist()
        self.answers = df['answer'].astype(str).tolist()

        #load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        #precompute embeddings for all questions
        self.question_embeddings = self.model.encode(self.questions)

    def embed(self, text):
        return self.model.encode([text])

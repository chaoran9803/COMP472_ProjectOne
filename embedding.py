import pandas as pd
from sentence_transformers import SentenceTransformer

def load_knowledge_base(csv_path = "knowledge_base.csv"):
    df = pd.read_csv(csv_path)
    return df["question"].tolist(), df["answer"].tolist()

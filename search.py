from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_best_answer(engine, user_query):
    # convert user query to embedding
    user_emb = engine.embed(user_query)

    # compute cosine similarity between user query and all questions
    scores = cosine_similarity(user_emb, engine.question_embeddings)[0]

    # find the index of the highest similarity score
    best_idx = np.argmax(scores)

    # return the best answer and similarity score
    return engine.answers[best_idx], scores[best_idx]
import re
import numpy as np
import math
import argparse
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.metrics.pairwise import cosine_similarity
from relation_extraction import *

st_model = SentenceTransformer('neuml/pubmedbert-base-embeddings')


def convert_relation2text(line):
    pattern_relation = r"^R\d+:\s*(.*?)\s*->\s*([^->]+)\s*->\s*(.*?)$"
    match = re.match(pattern_relation, line)
    if match:
        subject = match.group(1).lower().strip()
        object = match.group(3).lower().strip()
        relation = match.group(2).lower().strip()
        text = f"{subject} {relation} {object}"
        #print(text)
        return text
    else:
        return None


def read_textual_relations(input_file):
    input_text = read_content_from_file(input_file).split("Content returned:")[1]
    relations = []
    relations_text = [r for r in input_text.split("\n") if r]
    for row in relations_text:
        relations.append(convert_relation2text(row))
    return relations


def semantic_search(corpus_embeddings, queries_embeddings):  # corpus=encounter, query=explanation
    scores_90p = []
    for q, query_emb in enumerate(queries_embeddings):
        similarities_scores = st_model.similarity(query_emb, corpus_embeddings)[0]
        score = np.percentile(similarities_scores, 90)
        scores_90p.append(score)
    return scores_90p


if __name__ == "__main__":
    # Read the relations extracted from the encounter and explanation to compute semantic alignment.
    parser = argparse.ArgumentParser(description='Encounter data and explanation for semantic alignment computation.')
    parser.add_argument('encounter', type=str, help='Path to the file containing the relations of the encounter.')
    parser.add_argument('explanation', type=str, help='Path to the file containing the relations of the explanation.')
    args = parser.parse_args()
    encounter = read_textual_relations(args.encounter)
    explanation = read_textual_relations(args.explanation)

    # Generation of embeddings for extracted relations
    encounter_embeddings = st_model.encode(encounter, convert_to_tensor=True)
    explanation_embeddings = st_model.encode(explanation, convert_to_tensor=True)

    # Computation of the alignment score
    # >semantic_search_score: 90-percentile of the resulting similarity distribution for a relation of the explanation
    semantic_search_scores = semantic_search(encounter_embeddings, explanation_embeddings)
    # >Average of the semantic search scores for specific relations obtained from the explanation
    print(f"Semantic alignment score: {np.mean(semantic_search_scores):.4f}")




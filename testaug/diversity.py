import spacy

import numpy as np
import networkx as nx

from tqdm import tqdm
from nltk.util import ngrams
from fast_bleu import SelfBLEU

def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)


def self_bleu_n(sentences, n):
    weights = {
        k: 1 / k * np.ones(k) for k in range(2, n+1)
    }
    return np.mean(SelfBLEU([text.split() for text in sentences],
                   weights).get_score()[n])


def distinct_dependency_paths(sentences):
    nlp = spacy.load("en_core_web_sm")
    deps = set()

    for text in tqdm(sentences):
        doc = nlp(text)
        G = nx.DiGraph()

        edges = [(token.i, child.i, {"dependency": child.dep_}) for token in doc for child in token.children]
        G.add_edges_from(edges)

        # some sentences could have multiple roots
        roots = [n for n, d in G.in_degree() if d == 0]
        
        for root in roots:
            paths = [nx.shortest_path(G, root, node) 
                        for node in G if G.out_degree(node) == 0 and nx.has_path(G, root, node)]

            for path in paths:
                dep = ""
                for src, tgt in zip(path, path[1:]):
                    dep += "{} ".format(G[src][tgt]["dependency"])
                
                deps.add(dep.strip())
    
    return len(deps)


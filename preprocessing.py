import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")


def tokenize(sentence):
    return nlp(sentence.lower())


def lemma(word):
    return word.lemma_


def bag_of_words(sen_tok, total_words):
    sen_tok = [lemma(w) for w in sen_tok if not w.is_stop]

    bag = np.zeros(len(total_words), dtype=np.float32)
    for idx, w in enumerate(total_words):
        if w in sen_tok:
            bag[idx] = 1.0

    return bag

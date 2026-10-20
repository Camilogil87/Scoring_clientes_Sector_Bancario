# -*- coding: utf-8 -*-
import re

import numpy as np
from nltk.corpus import stopwords


def check_embeddings(embeddings):
    error_text = "Something wrong with your embeddings ('%s test isn't correct)."
    most_similar = embeddings.most_similar(
        positive=['woman', 'king'], negative=['man'])
    if len(most_similar) < 1 or most_similar[0][0] != 'queen':
        return error_text % "Most similar"

    doesnt_match = embeddings.doesnt_match(
        ['breakfast', 'cereal', 'dinner', 'lunch'])
    if doesnt_match != 'cereal':
        return error_text % "Doesn't match"

    most_similar_to_given = embeddings.most_similar_to_given(
        'music', ['water', 'sound', 'backpack', 'mouse'])
    if most_similar_to_given != 'sound':
        return error_text % "Most similar to given"

    return "¡Todo correcto!"


def question_to_vec_tests(question_to_vec, wv_embeddings):
    if (np.zeros(300) != question_to_vec('', wv_embeddings)).any():
        return "You need to return zero vector for empty question."
    if (np.zeros(300) != question_to_vec('thereisnosuchword', wv_embeddings)).any():
        return "You need to return zero vector for the question, which consists only unknown words."
    if (wv_embeddings['word'] != question_to_vec('word', wv_embeddings)).any():
        return "You need to check the corectness of your function."
    if ((wv_embeddings['I'] + wv_embeddings['am']) / 2 != question_to_vec('I am', wv_embeddings)).any():
        return "Your function should calculate a mean of word vectors."
    if (wv_embeddings['word'] != question_to_vec('thereisnosuchword word', wv_embeddings)).any():
        return "You should not consider words which embeddings are unknown."
    return "Test correctos."


def test_hits(hits_count):
    # *Evaluation example*
    # answers — dup_i
    answers = [
        "How does the catch keyword determine the type of exception that was thrown"]

    # candidates_ranking — the ranked sentences provided by our model
    candidates_ranking = [["How Can I Make These Links Rotate in PHP",
                           "How does the catch keyword determine the type of exception that was thrown",
                           "NSLog array description not memory address",
                           "PECL_HTTP not recognised php ubuntu"]]
    # dup_ranks — position of the dup_i in the list of ranks +1
    dup_ranks = [candidates_ranking[i].index(
        answers[i]) + 1 for i in range(len(answers))]

    # correct_answers — the expected values of the result for each k from 1 to 4
    correct_answers = [0, 1, 1, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return "Comprueba la definición de la función."

    # Other tests
    answers = ["How does the catch keyword determine the type of exception that was thrown",
               "Convert Google results object (pure js) to Python object"]

    # The first test: both duplicates on the first position in ranked list
    candidates_ranking = [["How does the catch keyword determine the type of exception that was thrown",
                           "How Can I Make These Links Rotate in PHP"],
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(
        answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [1, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return """
            Comprueba la definición de la función
            (test: los dos duplicados están en la primera posición del ranking.
            """
    # The second test: one candidate on the first position, another — on the
    # second
    candidates_ranking = [["How Can I Make These Links Rotate in PHP",
                           "How does the catch keyword determine the type of exception that was thrown"],
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(
        answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0.5, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return """
            Comprueba la definición de la función
            (test: un candidato en la primera posición, otro en la segunda).
            """

    # The third test: both candidates on the second position
    candidates_ranking = [["How Can I Make These Links Rotate in PHP",
                           "How does the catch keyword determine the type of exception that was thrown"],
                          ["WPF- How to update the changes in list item of a list",
                           "Convert Google results object (pure js) to Python object"]]
    dup_ranks = [candidates_ranking[i].index(
        answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return """
            Comprueba la definición de la función
            (test: los dos duplicados están en la segunda posición del ranking.
            """

    return "Tests correctos."


def test_dcg(dcg_score):
    # *Evaluation example*
    # answers — dup_i
    answers = [
        "How does the catch keyword determine the type of exception that was thrown"]

    # candidates_ranking — the ranked sentences provided by our model
    candidates_ranking = [["How Can I Make These Links Rotate in PHP",
                           "How does the catch keyword determine the type of exception that was thrown",
                           "NSLog array description not memory address",
                           "PECL_HTTP not recognised php ubuntu"]]
    # dup_ranks — position of the dup_i in the list of ranks +1
    dup_ranks = [candidates_ranking[i].index(
        answers[i]) + 1 for i in range(len(answers))]

    # correct_answers — the expected values of the result for each k from 1 to 4
    correct_answers = [0, 1 / (np.log2(3)), 1 / (np.log2(3)), 1 / (np.log2(3))]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return "Comprueba la función."

    # Other tests
    answers = ["How does the catch keyword determine the type of exception that was thrown",
               "Convert Google results object (pure js) to Python object"]

    # The first test: both duplicates on the first position in ranked list
    candidates_ranking = [["How does the catch keyword determine the type of exception that was thrown",
                           "How Can I Make These Links Rotate in PHP"],
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(
        answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [1, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return """
            Comprueba la definición de la función
            (test: los dos duplicados están en la primera posición del ranking.
            """
    # The second test: one candidate on the first position, another — on the
    # second
    candidates_ranking = [["How Can I Make These Links Rotate in PHP",
                           "How does the catch keyword determine the type of exception that was thrown"],
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(
        answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0.5, (1 + (1 / (np.log2(3)))) / 2]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return """
            Comprueba la definición de la función
            (test: un candidato en la primera posición, otro en la segunda).
            """

    # The third test: both candidates on the second position
    candidates_ranking = [["How Can I Make These Links Rotate in PHP",
                           "How does the catch keyword determine the type of exception that was thrown"],
                          ["WPF- How to update the changes in list item of a list",
                           "Convert Google results object (pure js) to Python object"]]
    dup_ranks = [candidates_ranking[i].index(
        answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0, 1 / (np.log2(3))]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return """
            Comprueba la definición de la función
            (test: los dos duplicados están en la segunda posición del ranking.
            """

    return "Tests correctos."


def test_rank_candidates(rank_candidates, wv_embeddings):
    questions = ['converting string to list', 'Sending array via Ajax fails']
    candidates = [['Convert Google results object (pure js) to Python object',
                   'C# create cookie from string and send it',
                   'How to use jQuery AJAX for an outside domain?'],
                  ['Getting all list items of an unordered list in PHP',
                   'WPF- How to update the changes in list item of a list',
                   'select2 not displaying search results']]
    results = [[(1, 'C# create cookie from string and send it'),
                (0, 'Convert Google results object (pure js) to Python object'),
                (2, 'How to use jQuery AJAX for an outside domain?')],
               [(0, 'Getting all list items of an unordered list in PHP'),
                (2, 'select2 not displaying search results'),
                (1, 'WPF- How to update the changes in list item of a list')]]
    for question, q_candidates, result in zip(questions, candidates, results):
        ranks = rank_candidates(question, q_candidates, wv_embeddings,)
        if not np.all(ranks == result):
            return "Comprueba la función."
    return "Test correctos."


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
GOOD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def text_prepare(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = GOOD_SYMBOLS_RE.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in STOPWORDS])
    return text.strip()

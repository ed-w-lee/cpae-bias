# Based off of https://github.com/chadaeun/weat_replication/ and https://github.com/hljames/compare-embedding-bias

import argparse
import json
import numpy as np
import random

try:
    from . import embedding
except:
    import embedding
from math import factorial
from sklearn.metrics.pairwise import cosine_similarity
from sympy.utilities.iterables import multiset_permutations
from tqdm import tqdm


def nCr(n, r):
    return factorial(n) // factorial(r) // factorial(n-r)


def embeds_from_list(embed, word_list):
    return np.array([embed[w] for w in word_list])


def weat_association(W, A, B):
    simWA = cosine_similarity(W, A)
    simWB = cosine_similarity(W, B)
    return np.mean(simWA, axis=-1) - np.mean(simWB, axis=-1)


def weat_score(X, Y, A, B):
    assoc_X = weat_association(X, A, B)
    assoc_Y = weat_association(Y, A, B)
    mean_diff = np.mean(assoc_X) - np.mean(assoc_Y)
    std = np.std(np.concatenate((assoc_X, assoc_Y), axis=0))
    return mean_diff / std


def weat_diff_assoc(X, Y, A, B):
    return np.sum(weat_association(X, A, B)) - np.sum(weat_association(Y, A, B))


def weat_p_value(X, Y, A, B):
    base_diff = weat_diff_assoc(X, Y, A, B)
    target_words = np.concatenate((X, Y), axis=0)

    N = len(target_words)
    idx = np.zeros(N, dtype=bool)
    idx[:N//2] = 1

    all_diffs = []
    expected_iters = nCr(N, N//2)
    if nCr(N, N//2) < 25000:
        for partition in tqdm(multiset_permutations(idx), total=expected_iters):
            partition = np.array(partition, dtype=bool)
            partition_X = target_words[partition]
            partition_Y = target_words[~partition]
            all_diffs.append(weat_diff_assoc(partition_X, partition_Y, A, B))
    else:
        for _ in tqdm(range(25000)):
            shuffle = np.random.shuffle(target_words)
            partition_X = target_words[:N//2]
            partition_Y = target_words[N//2:]
            all_diffs.append(weat_diff_assoc(partition_X, partition_Y, A, B))

    all_diffs = np.array(all_diffs)
    return np.sum(all_diffs > base_diff - 1e-12) / len(all_diffs)


def run_weat(embed, test):
    '''
    Runs a WEAT test. The embed should be a gensim w2v model, and the test should be formatted as:
    ```
    {
        'a_key': <A_KEY>,
        <A_KEY>: [words..],
        'b_key': <B_KEY>,
        <B_KEY>: [words..],
        'x_key': <X_KEY>,
        <X_KEY>: [words..],
        'y_key': <Y_KEY>,
        <Y_KEY>: [words..]
    }
    ```
    '''
    # re-format test dict
    word_lists = {
        'a': test[test['a_key']],
        'b': test[test['b_key']],
        'x': test[test['x_key']],
        'y': test[test['y_key']],
    }

    # filter found words
    word_pairs = {}
    for list_name, words in word_lists.items():
        words_filtered = list(
            filter(lambda x: x in embed and np.count_nonzero(embed[x]), words))

        print(f'{list_name} -- found {len(words_filtered)}/{len(words)}')
        word_pairs[list_name] = words_filtered

    X_len = len(word_pairs['x'])
    Y_len = len(word_pairs['y'])
    if X_len > Y_len:
        word_pairs['x'] = random.sample(word_pairs['x'], Y_len)
    elif Y_len > X_len:
        word_pairs['y'] = random.sample(word_pairs['y'], X_len)

    embeds = {k: embeds_from_list(embed, v) for k, v in word_pairs.items()}
    score = weat_score(embeds['x'], embeds['y'], embeds['a'], embeds['b'])
    p_val = weat_p_value(embeds['x'], embeds['y'], embeds['a'], embeds['b'])
    print(score, p_val)
    return score, p_val


if __name__ == "__main__":
    parse_args = argparse.ArgumentParser(
        description='Run WEAT tests based on given JSON')
    parse_args.add_argument('--tests', '-t', default='results/weat.json')
    parse_args.add_argument('--out', '-o', default='results/out.json')
    args = parse_args.parse_args()

    with open(args.tests, 'r') as fin:
        tests = json.load(fin)

    models = {
        'cpae': embedding.EmbedCpae(),
        'cpae-np': embedding.EmbedCpaeV2(),
        'cpae-np-pro': embedding.EmbedCpaeV3(),
        'd2v': embedding.EmbedDict2V(),
        'glove': embedding.EmbedGlove(),
        'w2v': embedding.EmbedWord2V()
    }

    results = {}
    for model_name, model in models.items():
        print('='*10, model_name, '='*10)
        results[model_name] = {}
        embed = model.get_gs()
        for test_name, test in tests.items():
            print(test_name)
            score, p_val = run_weat(embed, test)
            results[model_name][test_name] = {
                'score': float(score),
                'p_val': float(p_val)
            }

    with open(args.out, 'w+') as fout:
        json.dump(results, fout)

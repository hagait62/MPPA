# Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2019, Hagai Taitelbaum
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import numpy as np
import torch
from ..utils import get_nn_avg_dist, get_dict_path
import re
from collections import defaultdict

logger = getLogger()


def load_identical_char_dico(word2id1, word2id2, return_numpy=False):
    """
    Build a dictionary of identical character strings.
    """
    pairs = [(w1, w1) for w1 in word2id1.keys() if w1 in word2id2]
    if len(pairs) == 0:
        raise Exception("No identical character strings were found. "
                        "Please specify a dictionary.")

    logger.info("Found %i pairs of identical character strings." % len(pairs))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    if return_numpy: dico = np.empty([len(pairs),2],dtype=np.int64)
    else: dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico

def load_identical_num_dico(word2id1, word2id2, return_numpy=False):
    """
    Build a dictionary of identical character strings.
    """
    numeral_regex = re.compile('^[0-9]+$')
    src_numerals = {word for word in word2id1.keys() if numeral_regex.match(word) is not None}
    trg_numerals = {word for word in word2id2.keys() if numeral_regex.match(word) is not None}
    numerals = src_numerals.intersection(trg_numerals)
    pairs = [(w1, w1) for w1 in numerals]
    if len(pairs) == 0:
        raise Exception("No identical character strings were found. "
                        "Please specify a dictionary.")

    logger.info("Found %i pairs of identical character strings." % len(pairs))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    if return_numpy: dico = np.empty([len(pairs),2],dtype=np.int64)
    else: dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico

def load_dictionary(path, word2id1, word2id2,return_numpy=False):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
    print(path)
    assert os.path.isfile(path)
    included=[]
    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            assert line == line.lower()
            try:
                word1, word2 = line.rstrip().split()
            except:
                continue
            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
                included.append(i)
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    logger.info("Found %i pairs of words in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    if return_numpy: dico = np.empty([len(pairs),2],dtype=np.int64)
    else: dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico


def get_word_translation_accuracy(lang1, word2id1, emb1, lang2, word2id2, emb2, method, dico_eval, dicts_eval_path):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.
    """
    logger.info('Language pair %s-%s' % (lang1, lang2))
    path = get_dict_path(dico_eval, dicts_eval_path, lang1, lang2)
    dico = load_dictionary(path, word2id1, word2id2)
    dico = dico.cuda() if emb1.is_cuda else dico

    assert dico[:, 0].max() < emb1.size(0)
    assert dico[:, 1].max() < emb2.size(0)

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # nearest neighbors
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))

    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[dico[:, 0]][:, None] + average_dist2[None, :])

    else:
        raise Exception('Unknown method: "%s"' % method)

    results = []
    top_matches = scores.topk(100, 1, True)[1]
    matching = defaultdict(dict)
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1).cpu().numpy()
        # allow for multiple possible translations
        for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
            matching[src_id]['at_{}'.format(k)] = min(matching.get(src_id, {}).get('at_{}'.format(k),0) + _matching[i], 1)

        # evaluate precision@k
        precision_at_k = 100 * np.mean([matching[src_id]['at_{}'.format(k)] for src_id in matching.keys()])
        logger.info("%i source words - %s - Precision at k = %i: %f" %
                    (len(matching), method, k, precision_at_k))
        results.append(('precision_at_%i' % k, precision_at_k))
    return results
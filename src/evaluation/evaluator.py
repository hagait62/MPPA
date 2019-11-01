# Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2019, Hagai Taitelbaum
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from logging import getLogger
from copy import deepcopy
import numpy as np
import itertools

from .word_translation import get_word_translation_accuracy
from ..dico_builder import get_candidates, build_pairwise_dictionary
from src.utils import apply_mapping, get_dict_path
logger = getLogger()
import torch

class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.embs = trainer.embs
        self.lang_dico = trainer.lang_dico
        self.mapping = trainer.mapping
        self.params = trainer.params


    def word_translation(self, to_log):
        """
        Evaluation on word translation.
        """
        # mapped word embeddings
        from itertools import permutations
        for l1,l2 in permutations(self.params.langs, 2):
            torch.cuda.empty_cache()

            path = get_dict_path(self.params.dico_eval, self.params.dicts_eval_path, l1, l2)
            if not os.path.exists(path):
                logger.info('Warning: Test dictionary for {}-{} not exists. Skipping this pair'.format(l1, l2))
                continue

            src_emb = apply_mapping(self.mapping[l1],self.embs[l1].weight).data
            src_emb = src_emb.cuda() if self.params.cuda else src_emb.cpu()
            tgt_emb = apply_mapping(self.mapping[l2],self.embs[l2].weight).data
            src_emb = src_emb.cuda() if self.params.cuda else src_emb.cpu()


            for method in ['nn', 'csls_knn_10']:
                results = get_word_translation_accuracy(
                    l1, self.lang_dico[l1].word2id, src_emb,#.cuda(),
                    l2, self.lang_dico[l2].word2id, tgt_emb,#.cuda(),
                    method=method,
                    dico_eval=self.params.dico_eval, dicts_eval_path=self.params.dicts_eval_path)
                to_log.update([('%s-%s_%s-%s' % (k, method,l1,l2), v) for k, v in results])


    def dist_mean_cosine(self, to_log):
        """
        Mean-cosine model selection criterion.
        """
        mean_cosines = []
        # get normalized embeddings
        for l1,l2 in itertools.permutations(self.params.langs, 2):
            logger.info('compute mean cosine languages: {},{}'.format(l1,l2))
            # map embeddings to shared space
            src_emb = apply_mapping(self.mapping[l1],self.embs[l1].weight).data
            tgt_emb = apply_mapping(self.mapping[l2], self.embs[l2].weight).data

            # normalize mapped embeddings
            src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
            tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

            # build dictionary
            #for dico_method in ['nn', 'csls_knn_10']:
            for dico_method in ['csls_knn_10']:
                dico_build = 'S2T'
                dico_max_size = 10000
                # temp params / dictionary generation
                _params = deepcopy(self.params)
                _params.dico_method = dico_method
                _params.dico_build = dico_build
                _params.dico_threshold = 0
                _params.dico_max_rank = 10000
                _params.dico_min_size = 0
                _params.dico_max_size = dico_max_size
                s2t_candidates = get_candidates(src_emb, tgt_emb, _params)
                t2s_candidates = get_candidates(tgt_emb, src_emb, _params)
                dico = build_pairwise_dictionary(src_emb, tgt_emb, _params, s2t_candidates, t2s_candidates, True)
                # mean cosine
                if dico is None:
                    mean_cosine = -1e9
                else:
                    mean_cosine = (src_emb[dico[:dico_max_size, 0]] * tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
                    mean_cosine=mean_cosine.item()
                logger.info("Mean cosine (%s method, %s build, %i max size): %.5f"
                            % (dico_method, _params.dico_build, dico_max_size, mean_cosine))
                to_log['mean_cosine-%s-%s-%i_%s_%s' % (dico_method, _params.dico_build, dico_max_size,l1,l2)] = mean_cosine
                mean_cosines.append(mean_cosine)
        to_log['mean_cosine-%s-%s-%i' % (dico_method, _params.dico_build, dico_max_size)] = np.mean(mean_cosines)

    def all_eval(self, to_log):
        """
        Run all evaluations.
        """
        logger.info('Evaluating current mappings.')
        self.dist_mean_cosine(to_log)
        self.word_translation(to_log)
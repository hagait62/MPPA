# Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2019, Hagai Taitelbaum
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import scipy
import scipy.linalg
import itertools
import torch
from collections import defaultdict

from .utils import apply_mapping, export_embeddings, mult
from .dico_builder import build_multi_pairwise_dictionary
from .evaluation.word_translation import load_identical_char_dico, load_identical_num_dico, load_dictionary


logger = getLogger()


class Trainer(object):

    def __init__(self, embs, mapping, params):
        """
        Initialize trainer script.
        """
        self.embs = embs
        self.lang_dico = getattr(params, 'lang_dico', None)
        self.mapping = mapping
        self.params = params

        # best validation score
        self.best_valid_metric = -1e12

    def load_multi_pairwise_training_dico(self, dico_train, dicts_train_path):
        """
        Load training dictionary.
        """
        self.dico = defaultdict(dict)
        for src, tgt in itertools.permutations(self.params.langs,2):
            word2id1 = self.lang_dico[src].word2id
            word2id2 = self.lang_dico[tgt].word2id

            # identical character strings
            if dico_train == "identical_char":
                self.dico[src][tgt] = load_identical_char_dico(word2id1, word2id2, True)
            # identical numbers
            elif dico_train == 'identical_num':
                self.dico[src][tgt] = load_identical_num_dico(word2id1, word2id2, True)
            # use one of the provided dictionary
            elif dico_train == "default":
                filename = '%s-%s.0-5000.txt' % (src, tgt)
                self.dico[src][tgt] = load_dictionary(os.path.join(dicts_train_path, filename),word2id1, word2id2)
            elif dico_train == "MAT":
                filename = '%s-%s_extracred_after_MAT.txt' % (src, tgt)
                self.dico[src][tgt] = load_dictionary(os.path.join(dicts_train_path, filename), word2id1, word2id2)
            # dictionary provided by the user
            else:
                self.dico[src][tgt] = load_dictionary(dico_train, word2id1, word2id2, True)

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        embs = {lang: apply_mapping(self.mapping[lang],self.embs[lang].weight).data for lang in self.params.langs}
        embs = {lang: embs[lang] / embs[lang].norm(2, 1, keepdim=True).expand_as(embs[lang]) for lang in self.params.langs}
        self.dico = build_multi_pairwise_dictionary(embs, self.params)


    def init_mppa(self, T):

        init_lang = self.params.langs[0]
        logger.info('Initialization: align all languages to current shared space.'.format(init_lang))
        T[init_lang].copy_(torch.diag(torch.ones(T[init_lang].size(0))))
        used_langs = [init_lang]
        for lang in self.params.langs[1:]:
            M = self.get_efficient_M(lang,used_langs).cpu().numpy()
            U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
            T[lang].copy_(torch.from_numpy(U.dot(V_t)).type_as(T[lang]))
            used_langs.append(lang)

    def mppa(self, init):

        self.compute_cross_correlation()
        T = {lang: self.mapping[lang].weight.data for lang in self.params.langs}
        if init: self.init_mppa(T)
        scores = []
        for ii in range(self.params.epochs):
            for lang in self.params.langs:
                M = self.get_efficient_M(lang).cpu().numpy()
                assert M.shape == (300, 300), 'got shape: {}'.format(M.shape)
                U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
                T[lang].copy_(torch.from_numpy(U.dot(V_t)).type_as(T[lang]))
            score = self.evaluate_score()
            scores.append(score)
        logger.info('Finished MPPA for this refinement iteration.')
        return scores

    def compute_cross_correlation(self):
        logger.info('Pre-process: compute cross-correlation.')
        self.cc = {}
        for lang1,lang2 in itertools.permutations(self.params.langs,2):
            dico = self.dico[lang1][lang2]
            original_lang1_emb = self.embs[lang1].weight[dico[:, 0]]
            original_lang2_emb = self.embs[lang2].weight[dico[:, 1]]
            self.cc[(lang1, lang2)] = original_lang2_emb.t().mm(original_lang1_emb)


    def get_efficient_M(self,lang,lang_list = None):
        if lang_list is None:
            lang_list = self.params.langs
        M = 0
        for lang2 in lang_list:
            if lang2 == lang: continue
            else:
                #M += apply_mapping(self.mapping[lang2],self.cc[lang,lang2].t())
                M += mult(self.mapping[lang2].weight.data,self.cc[lang,lang2])
        #return M.t()
        return M


    def evaluate_score(self):
        score = 0
        n_elements = 0
        for l1,l2 in itertools.combinations(self.params.langs, 2):
            dico = self.dico[l1][l2]
            l1_univ_emb = apply_mapping(self.mapping[l1],self.embs[l1].weight[dico[:, 0]])
            l2_univ_emb = apply_mapping(self.mapping[l2],self.embs[l2].weight[dico[:, 1]])
            diff = l1_univ_emb-l2_univ_emb
            score += diff.norm(2,1).sum().item()
            n_elements += len(dico)
        return score/n_elements


    def save_best(self, to_log, metric, final_results):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            self.update_results(to_log, final_results)
            # save the mapping
            #W = {lang: self.mapping[lang].weight.data.cpu().numpy() for lang in self.params.langs}
            #path = {lang: os.path.join(self.params.exp_path, 'best_mapping.{}.pth'.format(lang)) for lang in self.params.langs}
            for lang in self.params.langs:
                W = self.mapping[lang].weight.data.cpu().numpy()
                path = os.path.join(self.params.exp_path, 'best_mapping.{}.pth'.format(lang))
                #logger.info('* Saving the mapping to %s ...' % path[lang])
                logger.info('* Saving the mapping to %s ...' % path)
                #torch.save(W[lang], path[lang])
                torch.save(W, path)

    def update_results(self, to_log, final_results):
        for src, tgt in itertools.permutations(self.params.langs, 2):
            final_results[('{}-{}'.format(src, tgt))] = to_log.get('precision_at_1-csls_knn_10_%s-%s' % (src, tgt), 0)

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = {lang: os.path.join(self.params.exp_path, 'best_mapping.{}.pth'.format(lang)) for lang in self.params.tgt_lang+[self.params.src_lang]}
        # reload the model
        for lang in self.params.tgt_lang+[self.params.src_lang]:
            to_reload = torch.from_numpy(torch.load(path[lang]))
            W = self.mapping[lang].weight.data
            logger.info('* Reloading the best model from %s ...' % path[lang])
            assert to_reload.size() == W.size()
            W.copy_(to_reload.type_as(W))


    def export(self):
        """
        Export embeddings.
        """
        #params = self.params
        # load all embeddings
        logger.info("Reloading all embeddings for mapping ...")

        src_emb = self.mapping[self.params.src_lang](self.src_emb.weight).data
        tgt_emb = {lang: self.mapping[lang](self.tgt_emb[lang].weight).data for lang in self.params.tgt_lang}
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = {lang: tgt_emb[lang] / tgt_emb[lang].norm(2, 1, keepdim=True).expand_as(tgt_emb[lang]) for lang in self.params.tgt_lang}
        export_embeddings(src_emb.cpu().numpy(), {lang: tgt_emb[lang].cpu().numpy() for lang in self.params.tgt_lang}, self.params)
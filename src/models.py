# Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2019, Hagai Taitelbaum
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn

from .utils import normalize_embeddings
from .load_embeddings import load_embeddings


def build_model(params):
    """
    Build all components of the model.
    """
    params.lang_dico = {}
    embs = {}
    # embeddings
    assert len(params.embs) == len(params.langs)
    for lang, emb in zip(params.langs,params.embs):
        dico, _emb = load_embeddings(lang, emb, params)
        params.lang_dico[lang] = dico
        embs[lang] = nn.Embedding(len(dico), params.emb_dim, sparse=True)
        embs[lang].weight.data.copy_(_emb)
        embs[lang].weight.requires_grad = False

    # mapping for each target language
    mapping = {lang: nn.Linear(params.emb_dim, params.emb_dim, bias=False) for lang in params.langs}
    for lang in params.langs:
        mapping[lang].weight.requires_grad=False
    if getattr(params, 'map_id_init', True):
        for lang in params.langs:
            mapping[lang].weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))

    # cuda
    if params.cuda:
        for lang in params.langs:
            mapping[lang].cuda()

    # normalize embeddings
    for lang in params.langs:
        normalize_embeddings(embs[lang].weight.data, params.normalize_embeddings)

    return embs, mapping
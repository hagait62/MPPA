# Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2019, Hagai Taitelbaum
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import argparse
from collections import OrderedDict
from itertools import permutations
import torch

from src.utils import bool_flag, initialize_exp, save_scores
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator


VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'

# main
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="results", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="", help="Export embeddings after training (txt / pth)")

# data
parser.add_argument("--langs", nargs='+', default=['en','de','fr', 'es', 'it', 'pt'], help="All languages")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--dicts_eval_path", type=str, default='./data/dictionaries/', help="path to evaluation dictionaries")
parser.add_argument("--dicts_train_path", type=str, default='./data/MAT_extracted_dictionaries/', help="path to training dictionaries")
#mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
# training epochs
parser.add_argument('--epochs', type=int, default=10, help="epochs for MPPA in each refinement iteration")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_train", type=str, default="default", help="Path to training dictionary (default: use identical character strings)")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--embs", nargs='+', type=str, default='', help="Reload embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="renorm,center", help="Normalize embeddings before training")



# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert params.dico_train in ["identical_char", "default","identical_num","MAT"] or os.path.isfile(params.dico_train)
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
assert all(os.path.isfile(emb) for emb in params.embs)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]
assert len(params.langs) == len(params.embs)

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
embs, mapping = build_model(params)
trainer = Trainer(embs, mapping, params)
evaluator = Evaluator(trainer)
final_results = {'{}-{}'.format(src, tgt):0 for src, tgt in permutations(params.langs, 2)}

# load a training dictionary. if a dictionary path is not provided, use a default
# one ("default") or create one based on identical character strings ("identical_char")
trainer.load_multi_pairwise_training_dico(params.dico_train, params.dicts_train_path)

"""
Learning loop for Multilingual Pairwise Procrustes Analysis (MPPA)
"""
for n_iter in range(params.n_refinement):

    logger.info('Starting iteration %i...' % n_iter)

    # build a dictionary from aligned embeddings (unless
    # it is the first iteration and we use the init one)
    if n_iter > 0 or not hasattr(trainer, 'dico'):
        trainer.build_dictionary()

    # apply MPPA
    scores = trainer.mppa(n_iter == 0)
    save_scores(scores, params.exp_path, n_iter)
    # embeddings evaluation
    to_log = OrderedDict({'n_iter': n_iter})
    evaluator.all_eval(to_log)

    # JSON log / save best model / end of epoch
    logger.info("__log__:%s" % json.dumps(to_log))
    trainer.save_best(to_log, VALIDATION_METRIC, final_results)
    logger.info('End of iteration %i.\n\n' % n_iter)

logger.info('\n\nFinal results (chosen by the best validation metric. '
            'Language pairs without available evaluation dictionary get 0 p@1):\n'
            '{}'.format(final_results))

# export embeddings
if params.export:
    trainer.reload_best()
    trainer.export()
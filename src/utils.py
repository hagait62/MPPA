# Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2019, Hagai Taitelbaum
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import io
import sys
import pickle
import random
import argparse
import subprocess
import numpy as np
import torch
from logging import getLogger

from .logger import create_logger


MAIN_DUMP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dumped')


logger = getLogger()

# load Faiss if available (dramatically accelerates the nearest neighbor search)
try:
    import faiss
    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False


def initialize_exp(params):
    """
    Initialize experiment.
    """
    if getattr(params, 'seed', -1) == -1:
        params.seed = np.random.randint(1e8)
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if params.cuda:
        torch.cuda.manual_seed(params.seed)

    # dump parameters
    params.exp_path = get_exp_path(params)
    with io.open(os.path.join(params.exp_path, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    # create logger
    logger = create_logger(os.path.join(params.exp_path, 'train.log'), vb=params.verbose)
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s' % params.exp_path)
    return logger


def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    if FAISS_AVAILABLE:
        emb = emb.cpu().numpy()
        query = query.cpu().numpy()
        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        distances, _ = index.search(query, knn)
        return distances.mean(1)
    else:
        bs = 1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cpu())
        all_distances = torch.cat(all_distances)
        return all_distances.numpy()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")

def get_exp_path(params):
    """
    Create a directory to store the experiment.
    """
    # create the main dump path if it does not exist
    exp_folder = MAIN_DUMP_PATH if params.exp_path == '' else params.exp_path
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % exp_folder, shell=True).wait()
    assert params.exp_name != ''
    exp_folder = os.path.join(exp_folder, params.exp_name)
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % exp_folder, shell=True).wait()
    if params.exp_id == '':
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        while True:
            exp_id = ''.join(random.choice(chars) for _ in range(10))
            exp_path = os.path.join(exp_folder, exp_id)
            if not os.path.isdir(exp_path):
                break
    else:
        exp_path = os.path.join(exp_folder, params.exp_id)
        assert not os.path.isdir(exp_path), exp_path
    # create the dump folder
    if not os.path.isdir(exp_path):
        subprocess.Popen("mkdir %s" % exp_path, shell=True).wait()
    return exp_path

def normalize_embeddings(emb, types, mean=None):
    """
    Normalize embeddings by their norms / recenter them.
    """
    for t in types.split(','):
        if t == '':
            continue
        if t == 'center':
            if mean is None:
                mean = emb.mean(0, keepdim=True)
            emb.sub_(mean.expand_as(emb))
        elif t == 'renorm':
            emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))
        else:
            raise Exception('Unknown normalization type: "%s"' % t)
    return mean.cpu() if mean is not None else None

def normalize_string(types):
    norm_string = ''
    for t in types.split(','):
        if t == '':
            norm_string += 'XX'
            return norm_string
        if t == 'center':
            norm_string += 'C'
        elif t == 'renorm':
            norm_string += 'R'
        else:
            raise Exception('Unknown normalization type: "%s"' % t)
    return norm_string

def export_embeddings(src_emb, tgt_emb, params):
    """
    Export embeddings to a text file.
    """
    if params.export == "txt":
        src_id2word = params.src_dico.id2word
        tgt_id2word = {lang: params.tgt_dico[lang].id2word for lang in params.tgt_lang}
        n_src = len(src_id2word)
        n_tgt = {lang: len(tgt_id2word[lang]) for lang in params.tgt_lang}
        dim = src_emb.shape[1]
        src_path = os.path.join(params.exp_path, 'vectors-%s.txt' % params.src_lang)
        tgt_path = {lang: os.path.join(params.exp_path, 'vectors-%s.txt' % lang) for lang in params.tgt_lang}
        # source embeddings
        logger.info('Writing source embeddings to %s ...' % src_path)
        with open(src_path, 'w') as f:
            f.write("%i %i\n" % (n_src, dim))
            for i in range(len(src_id2word)):
                f.write("%s %s\n" % (src_id2word[i], " ".join(str(x) for x in src_emb[i])))
        # target embeddings
        for lang in params.tgt_lang:
            logger.info('Writing target embeddings to %s ...' % tgt_path[lang])
            with open(tgt_path[lang], 'w') as f:
                f.write("%i %i\n" % (n_tgt[lang], dim))
                for i in range(len(tgt_id2word[lang])):
                    f.write("%s %s\n" % (tgt_id2word[lang][i], " ".join(str(x) for x in tgt_emb[lang][i])))

    if params.export == "pth":
        src_path = os.path.join(params.exp_path, 'vectors-%s.pth' % params.src_lang)
        tgt_path = {lang: os.path.join(params.exp_path, 'vectors-%s.pth' % lang) for lang in params.tgt_lang}
        logger.info('Writing source embeddings to %s ...' % src_path)
        torch.save({'dico': params.src_dico, 'vectors': src_emb}, src_path)
        for lang in params.tgt_lang:
            logger.info('Writing target embeddings to %s ...' % tgt_path[lang])
            torch.save({'dico': params.tgt_dico[lang], 'vectors': tgt_emb[lang]}, tgt_path[lang])

def save_scores(data,path,n_iter):
    with open(os.path.join(path,'scores_iter_{}.txt'.format(n_iter)),'w') as f:
        f.write('\n'.join([str(x) for x in data]))

def apply_mapping(mapping,input,device=None):
    if device is None:
        device = mapping.weight.device
    mapping = mapping.to(device)
    return mapping(input.to(device))

def mult(m1, m2, device=None):
    if device is None:
        device = m1.device
    m1 = m1.to(device)
    return m1.mm(m2.to(device))

def get_dict_path(dico_eval, dicts_eval_path, lang1, lang2):
    if dico_eval == 'default':
        return os.path.join(dicts_eval_path, '%s-%s.5000-6500.txt' % (lang1, lang2))
    else:
        return dico_eval
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to make and save model predictions on an input dataset."""

import sys

sys.path.append('.')
import os
import time
import torch
import argparse
import logging

try:
    import ujson as json
except ImportError:
    import json

from tqdm import tqdm
from predictor import Predictor
from multiprocessing import cpu_count

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, default=None,
                    help='SQuAD-like dataset to evaluate on')
parser.add_argument('--model', type=str, default=None,
                    help='Path to model to use')
parser.add_argument('--embedding-file', type=str, default=None,
                    help=('Expand dictionary to use all pretrained '
                          'embeddings in this file.'))
parser.add_argument('--char-embedding-file', type=str, default=None,
                    help=('Expand dictionary to use all pretrained '
                          'char embeddings in this file.'))
parser.add_argument('--out-dir', type=str, default='data/predict',
                    help=('Directory to write prediction file to '
                          '(<dataset>-<model>.preds)'))
parser.add_argument('--num-workers', type=int, default=int(cpu_count() / 2),
                    help='Number of CPU processes (for tokenizing, etc)')
parser.add_argument('--no-cuda', action='store_true',
                    help='Use CPU only')
parser.add_argument('--gpu', type=int, default=-1,
                    help='Specify GPU device id to use')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Example batching size')
parser.add_argument('--top-n', type=int, default=1,
                    help='Store top N predicted spans per example')
parser.add_argument('--official', type=bool, default=True,
                    help='Only store single top span instead of top N list')
args = parser.parse_args()
t0 = time.time()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

predictor = Predictor(
    args.model,
    normalize=True,
    embedding_file=args.embedding_file,
    char_embedding_file=args.char_embedding_file,
    num_workers=args.num_workers,
)
if args.cuda:
    predictor.cuda()

# ------------------------------------------------------------------------------
# Read in dataset and make predictions.
# ------------------------------------------------------------------------------


examples = []
qids = []
with open(args.dataset) as f:
    data = json.load(f)['data']
    for paragraph in data:
        context = paragraph['story']
        questions = paragraph['questions']
        for qa in questions:
            qids.append(paragraph['id'] + '-' + str(qa['turn_id']))
            examples.append((context, qa['input_text']))

results = {}
for i in tqdm(range(0, len(examples), args.batch_size)):
    predictions = predictor.predict_batch(
        examples[i:i + args.batch_size], top_n=args.top_n
    )
    for j in range(len(predictions)):
        # Official eval expects just a qid --> span
        if args.official:
            results[qids[i + j]] = predictions[j][0][0]
            results[qids[i + j] + '-yesno'] = predictions[j][0][1]

        # Otherwise we store top N and scores for debugging.
        else:
            results[qids[i + j]] = [(p[0], float(p[1])) for p in predictions[j]]

model = os.path.splitext(os.path.basename(args.model or 'default'))[0]
basename = os.path.splitext(os.path.basename(args.dataset))[0]
outfile = os.path.join(args.out_dir, basename + '-' + model + '.preds')
if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)
logger.info('Writing results to %s' % outfile)
with open(outfile, 'w') as f:
    json.dump(results, f)

logger.info('Total time: %.2f' % (time.time() - t0))

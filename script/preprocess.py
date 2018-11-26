#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Preprocess the SQuAD dataset for training."""

import sys

sys.path.append('.')
import argparse
import os

try:
    import ujson as json
except ImportError:
    import json
import time

from multiprocessing import Pool, cpu_count
from multiprocessing.util import Finalize
from functools import partial
from spacy_tokenizer import SpacyTokenizer

# ------------------------------------------------------------------------------
# Tokenize + annotate.
# ------------------------------------------------------------------------------

TOK = None
ANNTOTORS = {'lemma', 'pos', 'ner'}


def init():
    global TOK
    TOK = SpacyTokenizer(annotators=ANNTOTORS)
    Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': tokens.words(),
        'chars': tokens.chars(),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
    }
    return output


# ------------------------------------------------------------------------------
# Process dataset examples
# ------------------------------------------------------------------------------


def load_dataset(path):
    """Load json file and store fields separately."""
    with open(path) as f:
        data = json.load(f)['data']
    output = {'qids': [], 'questions': [], 'answers': [],
              'contexts': [], 'qid2cid': [], 'story_id': []}
    for paragraph in data:
        output['contexts'].append(paragraph['story'])
        # output['story_id'].append(paragraph['id'])
        questions = paragraph["questions"]
        answers = [[answer] for answer in paragraph["answers"]]
        if "addtional_answers" in paragraph:
            additional_answers = paragraph["additional_answers"]
            for key in additional_answers:
                for index, ans in enumerate(additional_answers[key]):
                    answers[index].append(ans)
        for question in questions:
            output['qids'].append(paragraph['id'] + str(question["turn_id"]))
            output['questions'].append(question["input_text"])
            output['qid2cid'].append(len(output['contexts']) - 1)
        for answer in answers:
            output['answers'].append(answer)
    return output


def find_answer(offsets, begin_offset, end_offset):
    """Match token offsets with the char begin/end offsets of the answer."""
    # some span_text in coqa have leading blanks which should be removed before this pre-process
    # otherwise it won't find the start
    start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]
    end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]
    assert (len(start) <= 1)
    assert (len(end) <= 1)
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]
    else:
        if len(start) == 0:
            start = []
            for i, tok in enumerate(offsets):
                if tok[0] <= begin_offset <= tok[1]:
                    start.append(i)
                    break
        if len(end) == 0:
            end = []
            for i, tok in enumerate(offsets):
                if tok[0] <= end_offset <= tok[1]:
                    end.append(i)
                    break
        if len(start) == 1 and len(end) == 1:
            return start[0], end[0]
        raise RuntimeError("debug here:\n begin_offset: %d\n end_offset: %d\n len(start) = %d\nlen(end) = %d\n" % (
            begin_offset, end_offset, len(start), len(end)))


def num_leading_blanks(span_text):
    num = 0
    for i in range(len(span_text)):
        if span_text[i] == ' ':
            num += 1
        else:
            return num


def process_dataset(data, tokenizer, workers=None):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""
    make_pool = partial(Pool, workers, initializer=init)

    workers = make_pool(initargs=())
    q_tokens = workers.map(tokenize, data['questions'])
    workers.close()
    workers.join()

    workers = make_pool(initargs=())
    c_tokens = workers.map(tokenize, data['contexts'])
    workers.close()
    workers.join()

    for idx in range(len(data['qids'])):
        question = q_tokens[idx]['words']
        question_char = q_tokens[idx]['chars']
        qlemma = q_tokens[idx]['lemma']
        qpos = q_tokens[idx]['pos']
        qner = q_tokens[idx]['ner']

        document = c_tokens[data['qid2cid'][idx]]['words']
        document_char = c_tokens[data['qid2cid'][idx]]['chars']
        offsets = c_tokens[data['qid2cid'][idx]]['offsets']
        clemma = c_tokens[data['qid2cid'][idx]]['lemma']
        cpos = c_tokens[data['qid2cid'][idx]]['pos']
        cner = c_tokens[data['qid2cid'][idx]]['ner']

        ans_tokens = []
        span_tokens = []
        yesno = 'x'
        if len(data['answers']) > 0:
            for ans in data['answers'][idx]:
                input_text = ans['input_text'].strip()
                span_text = ans['span_text']
                begin = span_text.find(input_text)
                # find the answer
                r_input_text = input_text.replace('\n', '').lower()
                if r_input_text == 'yes' or r_input_text == 'no':
                    if r_input_text == 'yes':
                        yesno = 'y'
                    else:
                        yesno = 'n'
                    leading_blanks = num_leading_blanks(span_text)
                    start = ans['span_start'] + leading_blanks
                    end = start + len(span_text.strip().replace('\n', ''))
                    found = find_answer(offsets, start, end)
                elif r_input_text == 'unknown':
                    found = (0, 0)
                else:
                    if begin != -1:
                        found = find_answer(offsets,
                                            ans['span_start'] + begin,
                                            ans['span_start'] + begin + len(input_text))
                    else:
                        leading_blanks = num_leading_blanks(span_text)
                        start = ans['span_start'] + leading_blanks
                        end = start + len(span_text.strip().replace('\n', ''))
                        found = find_answer(offsets, start, end)
                if found:
                    ans_tokens.append(found)
                else:
                    # ans_tokens.append((0, 0))
                    raise RuntimeError(
                        "can't find the answer tokens, span_text = \n%s\n span_start = %d\n input_text = %s" % (
                            span_text, ans['span_start'], input_text))
                # find the evidence
                if span_text.strip().replace('\n', '').lower() == 'unknown':
                    span_tokens.append((0, 0))
                else:
                    leading_blanks = num_leading_blanks(span_text)
                    start = ans['span_start'] + leading_blanks
                    end = start + len(span_text.strip().replace('\n', ''))
                    span = find_answer(offsets, start, end)
                    if span:
                        span_tokens.append(span)
                    else:
                        raise RuntimeError(
                            "can't find the span tokens, span_text = \n%s\n span_start = %d\n input_text = %s" % (
                                span_text, start, input_text))
        yield {
            'id': data['qids'][idx],
            'question': question,
            'question_char': question_char,
            'document': document,
            'document_char': document_char,
            'offsets': offsets,
            'answers': ans_tokens,
            'span': span_tokens,
            'yesno': yesno,
            'qlemma': qlemma,
            'qpos': qpos,
            'qner': qner,
            'clemma': clemma,
            'cpos': cpos,
            'cner': cner,
        }


# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Path to SQuAD data directory')
parser.add_argument('out_dir', type=str, help='Path to output file dir')
parser.add_argument('--split', type=str, help='Filename for train/dev split')
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--tokenizer', type=str, default='spacy')
args = parser.parse_args()

t0 = time.time()

in_file = os.path.join(args.data_dir, args.split + '.json')
print('Loading dataset %s' % in_file, file=sys.stderr)
dataset = load_dataset(in_file)

out_file = os.path.join(
    args.out_dir, '%s-processed-%s.txt' % (args.split, args.tokenizer)
)
print('Will write to file %s' % out_file, file=sys.stderr)
with open(out_file, 'w') as f:
    for ex in process_dataset(dataset, args.tokenizer, args.num_workers):
        f.write(json.dumps(ex) + '\n')
print('Total time: %.4f (s)' % (time.time() - t0))

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import argparse
import os
import csv
import glob
import json
import gzip
import sys
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator

import numpy as np
import torch
from torch import Tensor as T
from torch import nn

from dpr.data.qa_validation import calculate_matches
from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
from dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer, CustomIndexer

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer, index: DenseIndexer, is_colbert: bool):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index
        self.is_colbert = is_colbert

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):
                all_q = []
                for q in questions[batch_start:batch_start + bsz]:
                    q = ' '.join(['question:' , q])
                    all_q.append(q)

                #batch_token_tensors = [self.tensorizer.text_to_tensor(q) for q in
                #                       questions[batch_start:batch_start + bsz]]
                batch_token_tensors = [self.tensorizer.text_to_tensor(q, max_length=40) for q in all_q]
                #print(self.tensorizer.tokenizer.decode(list(batch_token_tensors[0].cpu().numpy())))

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info('Encoded queries %d', len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info('Total encoded queries tensor %s', query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def index_encoded_data(self, vector_files: List[str], buffer_size: int = 50000, memmap: np.memmap = False):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        if memmap is not None:
            memmap_pos = 0
        for i, item in enumerate(iterate_encoded_files(vector_files)):
            db_id, doc_vector = item
            #print('iter', db_id, doc_vector.shape)
            buffer.append((db_id, doc_vector))
            if 0 < buffer_size  and len(buffer) >= buffer_size:
                if memmap is not None:
                    vectors = np.vstack([t[1][None] for t in buffer])
                    memmap[memmap_pos:(memmap_pos+vectors.shape[0])] = vectors
                    memmap_pos += vectors.shape[0]
                self.index.index_data(buffer, self.is_colbert)
                buffer = []
        self.index.index_data(buffer, self.is_colbert)
        if memmap is not None:
            vectors = np.vstack([t[1][None] for t in buffer])
            memmap[memmap_pos:(memmap_pos+vectors.shape[0])] = vectors

        logger.info('Data indexing completed.')

        return memmap

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100, is_colbert: bool=False) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        if is_colbert:
            results = self.index.search_knn_colbert(query_vectors, top_docs)
        else:
            results = self.index.search_knn(query_vectors, top_docs) 
        logger.info('index search time: %f sec.', time.time() - time0)
        return results

    def colbert_search(self, query_vectors: np.array, memmap: np.memmap, top_ids: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        all_idx, all_scores, result = [], [], []
        all_docs, all_scores = [], []
        for i in range(len(query_vectors)):
            q_top_ids = top_ids[i][0]
            q_top_ids = np.array([int(x) for x in q_top_ids])
            document_vectors = memmap[q_top_ids-1] #-1 to convert to memmap index
            idx, scores = self.exact_score(query_vectors[i], document_vectors)
            doc_ids = q_top_ids[idx[:top_docs]]
            scores = scores[:top_docs]
            doc_ids = [str(k) for k in doc_ids]
            result.append((doc_ids, list(scores)))
            #all_docs.append(list(doc_ids))
            #all_scores.append(list(scores))
        #all_docs = np.vstack(all_docs)
        #all_scores = np.vstack(all_scores)
        #return all_docs, all_scores
        return result

    def exact_score(self, qv, dv):
        scores = np.einsum('id,kjd->kij', qv, dv)
        scores = scores.max(axis=2)
        scores = scores.sum(axis=1)
        sorted_idx = np.argsort(-scores)
        sorted_scores = scores[sorted_idx]
        return sorted_idx, sorted_scores




def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter='\t')
        for i, row in enumerate(reader):
            question = row[0]
            try:
                answers = eval(row[1])
            except:
                print(i, row)
            yield question, answers


def validate(passages: Dict[object, Tuple[str, str]], answers: List[List[str]],
             result_ctx_ids: List[Tuple[List[object], List[float]]],
             workers_num: int, match_type: str) -> List[List[bool]]:
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return match_stats.questions_doc_hits


def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info('Reading data from: %s', ctx_file)
    if ctx_file.startswith(".gz"):
        with gzip.open(ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
    return docs


def save_results(passages: Dict[object, Tuple[str, str]], questions: List[str], answers: List[List[str]],
                 top_passages_and_scores: List[Tuple[List[object], List[float]]], per_question_hits: List[List[bool]],
                 out_file: str
                 ):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append({
            'question': q,
            'answers': q_answers,
            'ctxs': [
                {
                    'id': results_and_scores[0][c],
                    'title': docs[c][1],
                    'text': docs[c][0],
                    'score': scores[c],
                    'has_answer': hits[c],
                } for c in range(ctxs_num)
            ]
        })

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info('Saved results * scores  to %s', out_file)


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info('Reading file %s', file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector


def main(args):
    questions = []
    question_answers = []
    for i, ds_item in enumerate(parse_qa_csv_file(args.qa_file)):
        #if i == 10:
        #    break
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)
    if not args.encoder_model_type == 'hf_attention' and not args.encoder_model_type == 'colbert':
        saved_state = load_states_from_checkpoint(args.model_file)
        set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    if not args.encoder_model_type == 'hf_attention' and not args.encoder_model_type == 'colbert':
        encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16)
    encoder.eval()

    if not args.encoder_model_type == 'hf_attention' and not args.encoder_model_type == 'colbert':
        # load weights from the model file
        model_to_load = get_model_obj(encoder)
        logger.info('Loading saved model state ...')

        prefix_len = len('question_model.')
        question_encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                                  key.startswith('question_model.')}
        model_to_load.load_state_dict(question_encoder_state)
        vector_size = model_to_load.get_out_size()
        logger.info('Encoder vector_size=%d', vector_size)
    else:
        vector_size = 16

    index_buffer_sz = args.index_buffer

    if args.index_type == 'hnsw':
        index = DenseHNSWFlatIndexer(vector_size, index_buffer_sz)
        index_buffer_sz = -1  # encode all at once
    elif args.index_type == 'custom':
        index = CustomIndexer(vector_size, index_buffer_sz)
    else:
        index = DenseFlatIndexer(vector_size)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index, args.encoder_model_type=='colbert')


    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    input_paths = glob.glob(ctx_files_pattern)
    #logger.info('Reading all passages data from files: %s', input_paths)
    #memmap = retriever.index_encoded_data(input_paths, buffer_size=index_buffer_sz, memmap=args.encoder_model_type=='colbert')
    print(input_paths)
    index_path = "_".join(input_paths[0].split("_")[:-1])
    memmap_path = 'memmap.npy'
    print(args.save_or_load_index, os.path.exists(index_path), index_path)
    if args.save_or_load_index and os.path.exists(index_path+'.index.dpr'):
    #if False:
        retriever.index.deserialize_from(index_path)
        if args.encoder_model_type=='colbert':
            memmap = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(21015324, 250, 16))
        else:
            memmap = None
    else:
        logger.info('Reading all passages data from files: %s', input_paths)
        if args.encoder_model_type=='colbert':
            memmap = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(21015324, 250, 16))
        else:
            memmap = None
        retriever.index_encoded_data(input_paths, buffer_size=index_buffer_sz, memmap=memmap)
        if args.save_or_load_index:
            retriever.index.serialize(index_path)
    # get questions & answers



    questions_tensor = retriever.generate_question_vectors(questions)

    # get top k results
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs, is_colbert=args.encoder_model_type=='colbert')

    with open('approx_scores.pkl', 'wb') as f:
        pickle.dump(top_ids_and_scores, f)

    retriever.index.index.reset()
    if args.encoder_model_type=='colbert':
        logger.info('Colbert score') 
        top_ids_and_scores_colbert = retriever.colbert_search(questions_tensor.numpy(), memmap, top_ids_and_scores, args.n_docs)

    all_passages = load_passages(args.ctx_file)

    if len(all_passages) == 0:
        raise RuntimeError('No passages data found. Please specify ctx_file param properly.')


    with open('colbert_scores.pkl', 'wb') as f:
        pickle.dump(top_ids_and_scores_colbert, f)

    

    questions_doc_hits = validate(all_passages, question_answers, top_ids_and_scores, args.validation_workers,
                                  args.match)
    if args.encoder_model_type=='colbert':
        questions_doc_hits_colbert = validate(all_passages, question_answers, top_ids_and_scores_colbert, args.validation_workers,
                                  args.match)
        

    if args.out_file:
        save_results(all_passages, questions, question_answers, top_ids_and_scores, questions_doc_hits, args.out_file)
        if args.encoder_model_type=='colbert':
            save_results(all_passages, questions, question_answers, top_ids_and_scores_colbert, questions_doc_hits_colbert, args.out_file+'colbert')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--qa_file', required=True, type=str, default=None,
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--ctx_file', required=True, type=str, default=None,
                        help="All passages file in the tsv format: id \\t passage_text \\t title")
    parser.add_argument('--encoded_ctx_file', type=str, default=None,
                        help='Glob path to encoded passages (from generate_dense_embeddings tool)')
    parser.add_argument('--out_file', type=str, default=None,
                        help='output .tsv file path to write results to ')
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'],
                        help="Answer matching logic type")
    parser.add_argument('--n-docs', type=int, default=200, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--index_type", type=str, default='flat', help='Index type')
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')

    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)
    print_args(args)
    main(args)

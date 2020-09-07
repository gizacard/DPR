#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 FAISS-based index components for dense retriver
"""

import os
import logging
import pickle
from typing import List, Tuple

import faiss
import numpy as np

logger = logging.getLogger()


class DenseIndexer(object):

    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        #self.index_id_to_db_id = []
        self.index_id_to_db_id = np.empty((0), dtype=np.int64)
        self.index = None

    def index_data(self, data: List[Tuple[object, np.array]]):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        raise NotImplementedError

    def serialize(self, file: str):
        logger.info('Serializing index to %s', file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + '.index.dpr'
            meta_file = file + '.index_meta.dpr'

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, file: str):
        logger.info('Loading index from %s', file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + '.index.dpr'
            meta_file = file + '.index_meta.dpr'

        self.index = faiss.read_index(index_file)
        logger.info('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        new_ids = np.array(db_ids, dtype=np.int64)
        self.index_id_to_db_id = np.concatenate((self.index_id_to_db_id, new_ids), axis=0)
        #self.index_id_to_db_id.extend(db_ids)


class DenseFlatIndexer(DenseIndexer):

    def __init__(self, vector_sz: int, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: List[Tuple[object, np.array]], is_colbert: bool = False):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            if is_colbert:
                db_ids = []
                for t in data[i:i+self.buffer_size]:
                    db_ids.extend([int(t[0])]*t[1].shape[0])
            else:
                db_ids = [t[0] for t in data[i:i + self.buffer_size]]
                #vectors = [np.reshape(t[1], (1, -1)) for t in data[i:i + self.buffer_size]]
                #vectors = np.concatenate(vectors, axis=0)
            vectors = np.vstack([t[1] for t in data[i:i+self.buffer_size]]) #colbert
            self._update_id_mapping(db_ids)
            self.index.add(vectors)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info('Total data indexed %d', indexed_cnt)

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        scores, indexes = self.index.search(query_vectors, top_docs)
        # convert to external ids
        db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def search_knn_all(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        nqueries, nvectors, dim = query_vectors.shape 
        query_vectors = query_vectors.reshape(nqueries*nvectors, dim)
        scores, indexes = self.index.search(query_vectors, top_docs)
        # convert to external ids
        indexes = indexes.reshape(nqueries, nvectors, top_docs)
        scores = scores.reshape(nqueries, nvectors, top_docs)
        db_ids = self.index_id_to_db_id[indexes]
        result = []
        top_doc = []
        for i in range(len(scores)):
            max_scores = []
            ids = []
            example_ids = db_ids[i].reshape(-1)
            example_scores = scores[i].reshape(-1)
            unique_db_ids, unique_index = np.unique(example_ids, return_inverse=True)
            for j in range(len(unique_db_ids)):
                select_arr = (unique_index==j)
                select_scores = example_scores[select_arr]
                max_scores.append(np.max(select_scores))
                ids.append(unique_db_ids[j])
            max_scores = np.array(max_scores)
            ids = np.array(ids)
            idx = np.argsort(-np.array(max_scores))
            sorted_ids = ids[idx]
            sorted_scores = max_scores[idx]
            result.append((sorted_ids, sorted_scores))
            top_doc.append(sorted_ids)
        return top_doc


class DenseHNSWFlatIndexer(DenseIndexer):
    """
     Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(self, vector_sz: int, buffer_size: int = 50000, store_n: int = 512
                 , ef_search: int = 128, ef_construction: int = 200):
        super(DenseHNSWFlatIndexer, self).__init__(buffer_size=buffer_size)

        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWFlat(vector_sz + 1, store_n)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
        self.index = index
        self.phi = 0

    def index_data(self, data: List[Tuple[object, np.array]], is_colbert: bool):
        n = len(data)

        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.phi > 0:
            raise RuntimeError('DPR HNSWF index needs to index all data at once,'
                               'results will be unpredictable otherwise.')
        phi = 0
        for i, item in enumerate(data):
            id, doc_vector = item
            norms = (doc_vector ** 2).sum()
            phi = max(phi, norms)
        logger.info('HNSWF DotProduct -> L2 space phi={}'.format(phi))
        self.phi = 0

        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids = [t[0] for t in data[i:i + self.buffer_size]]
            vectors = [np.reshape(t[1], (1, -1)) for t in data[i:i + self.buffer_size]]

            norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
            aux_dims = [np.sqrt(phi - norm) for norm in norms]
            hnsw_vectors = [np.hstack((doc_vector, aux_dims[i].reshape(-1, 1))) for i, doc_vector in
                            enumerate(vectors)]
            hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)

            self._update_id_mapping(db_ids)
            self.index.add(hnsw_vectors)
            logger.info('data indexed %d', len(self.index_id_to_db_id))

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info('Total data indexed %d', indexed_cnt)

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:

        aux_dim = np.zeros(len(query_vectors), dtype='float32')
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        logger.info('query_hnsw_vectors %s', query_nhsw_vectors.shape)
        scores, indexes = self.index.search(query_nhsw_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def deserialize_from(self, file: str):
        super(DenseHNSWFlatIndexer, self).deserialize_from(file)
        # to trigger warning on subsequent indexing
        self.phi = 1


class IVFPQIndexer(DenseIndexer):
    """
     Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(self, vector_sz: int, buffer_size: int = 50000, nlist: int = 128, n_subquantizers: int = 64):
        super(IVFPQIndexer, self).__init__(buffer_size=buffer_size)

        #quantizer = faiss.IndexFlatIP(vector_sz)  # this remains the same
        #index = faiss.IndexIVFPQ(quantizer, vector_sz, nlist, n_subquantizers, 8)# faiss.METRIC_INNER_PRODUCT)
        #quantizer = faiss.IndexFlatIP(vector_sz)  # the other index
        #index = faiss.IndexIVFFlat(quantizer, vector_sz, nlist) 
        #index = faiss.index_factory(vector_sz, "PQ128x8", faiss.METRIC_INNER_PRODUCT)
        quantizer = faiss.IndexFlatIP(vector_sz)
        index = faiss.IndexIVFFlat(quantizer, vector_sz, 256, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = 8

        self.index = index

    #def index_data(self, data: List[Tuple[object, np.array]]):
    #    n = len(data)
    #    # indexing in batches is beneficial for many faiss index types
    #    for i in range(0, n, self.buffer_size):
    #        db_ids = [t[0] for t in data[i:i + self.buffer_size]]
    #        db_ids = [[t[0]] * t[1].size(0) for t in data[i:i + self.buffer_size]]
    #        vectors = np.vstack([t[1] for t in data[i:i+self.buffer_size]])
    #        vectors = [np.reshape(t[1], (1, -1)) for t in data[i:i + self.buffer_size]]
    #        #vectors = np.concatenate(vectors, axis=0)
    #        self._update_id_mapping(db_ids)
    #        if not self.index.is_trained:
    #            self.index.train(vectors)
    #        self.index.add(vectors)

    #    indexed_cnt = len(self.index_id_to_db_id)
    #    logger.info('Total data indexed %d', indexed_cnt)

    def index_data(self, data: List[Tuple[object, np.array]], is_colbert: bool):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types

        for i in range(0, n, self.buffer_size):
            if is_colbert:
                db_ids = []
                for t in data[i:i+self.buffer_size]:
                    db_ids.extend([int(t[0])]*t[1].shape[0])
            else:
                db_ids = [t[0] for t in data[i:i + self.buffer_size]]
            vectors = np.vstack([t[1] for t in data[i:i+self.buffer_size]]) #colbert
            self._update_id_mapping(db_ids)
            print(vectors.shape)
            if not self.index.is_trained:
                self.index.train(vectors)
            self.index.add(vectors)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info('Total data indexed %d', indexed_cnt)

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        scores, indexes = self.index.search(query_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result


    def search_knn_all(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        nqueries, nvectors, dim = query_vectors.shape 
        query_vectors = query_vectors.reshape(nqueries*nvectors, dim)
        scores, indexes = self.index.search(query_vectors, top_docs)
        # convert to external ids
        indexes = indexes.reshape(nqueries, nvectors, top_docs)
        scores = scores.reshape(nqueries, nvectors, top_docs)
        db_ids = self.index_id_to_db_id[indexes]
        result = []
        top_doc = []
        for i in range(len(scores)):
            max_scores = []
            ids = []
            example_ids = db_ids[i].reshape(-1)
            example_scores = scores[i].reshape(-1)
            unique_db_ids, unique_index = np.unique(example_ids, return_inverse=True)
            for j in range(len(unique_db_ids)):
                select_arr = (unique_index==j)
                select_scores = example_scores[select_arr]
                max_scores.append(np.max(select_scores))
                ids.append(unique_db_ids[j])
            max_scores = np.array(max_scores)
            ids = np.array(ids)
            idx = np.argsort(-np.array(max_scores))
            sorted_ids = ids[idx]
            sorted_scores = max_scores[idx]
            result.append((sorted_ids, sorted_scores))
            top_doc.append(sorted_ids)
        return top_doc

#!/bin/bash
#SBATCH --cpus-per-task=30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=gdpr
#SBATCH --output=/private/home/%u/DPR/run_dir/%A
#SBATCH --partition=dev
#SBATCH --mem=200GB
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append

#mp='/checkpoint/gizacard/qacache/retriever_trivia/30340124_100/checkpoint/step-15000/' #i1 bert ini trivia
mp='/checkpoint/gizacard/qacache/retriever_trivia/30336886_100/checkpoint/step-20000/' #i1 bm25 trivia
split=dev

~/anaconda3/envs/dpr/bin/python3 dense_retriever.py \
    --model_file $mp \
    --ctx_file data/wikipedia_split/psgs_w100.tsv \
    --qa_file data/retriever/qas/trivia-$split.csv \
    --encoded_ctx_file "data/trivia_bm25_i1/wiki_passages_*" \
    --out_file contexts/trivia_bm25_i1/$split.json \
    --n-docs 100 \
    --validation_workers 64 \
    --batch_size 128 \
    --encoder_model_type hf_attention \
    --index_buffer 500000 \

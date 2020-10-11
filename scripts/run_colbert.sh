#!/bin/bash
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=gdpr
#SBATCH --output=/private/home/%u/DPR/run_dir/%A
#SBATCH --partition=dev
#SBATCH --mem=700GB
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
mp='/checkpoint/gizacard/qacache/retriever_nq/29902766_100_16/checkpoint/step-35000' #16 dim colbert
split=test

~/anaconda3/envs/dpr/bin/python3 dense_retriever.py \
    --save_or_load_index \
    --model_file $mp \
    --ctx_file data/wikipedia_split/psgs_w100.tsv \
    --qa_file data/retriever/qas/nq-$split.csv \
    --encoded_ctx_file "data/wi_colbert16/wiki_passages_*" \
    --out_file contexts/nq_colbert16/$split.json \
    --n-docs 100 \
    --validation_workers 64 \
    --batch_size 128 \
    --encoder_model_type colbert \
    --index_buffer 1000000 \
    --index_type custom \

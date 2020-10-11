#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00
#SBATCH --job-name=gdpr
#SBATCH --output=/private/home/%u/DPR/run_dir/%A
#SBATCH --partition=dev
#SBATCH --mem=120GB
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append

#wikipath = data/wikipedia_split/psgs_w100.tsv
#wikipath = '/private/home/gizacard/qacache/wiki_file.tsv'
#mp='/checkpoint/gizacard/qacache/retriever_nq/28353318_100/'
#mp='/checkpoint/gizacard/qacache/retriever_nq/28506165_75/'
#mp='/checkpoint/gizacard/qacache/retriever_nq/28353318_100/'
mp='/checkpoint/gizacard/qacache/retriever_nq/28750188_100'
#mp='checkpoint/retriever/multiset/bert-base-encoder.cp' 
mp='/checkpoint/gizacard/qacache/retriever_nq/29322506_100/checkpoint/step-23500'
mp='/checkpoint/gizacard/qacache/retriever_nq/29326592_100/checkpoint/step-22500'


~/anaconda3/envs/dpr/bin/python3 generate_dense_embeddings.py \
    --model_file $mp \
    --ctx_file data/wikipedia_split/psgs_w100.tsv \
    --shard_id 71 \
    --num_shards 100 \
    --out_file data/wi_mean_i1_kl_d01/wiki_passages \
    --batch_size 256 \
    --encoder_model_type hf_attention \

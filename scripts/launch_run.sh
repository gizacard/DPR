#!/bin/bash
#SBATCH --cpus-per-task=30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=gdpr
#SBATCH --output=/private/home/%u/DPR/run_dir/%A
#SBATCH --partition=dev
#SBATCH --mem=700GB
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append

#gp="data/wikipedia_index_attn_w100/wiki_passages_*"
#gp="data/wikipedia_index_attn_w100/wiki_passages_*"
#mp='/checkpoint/gizacard/qacache/retriever_nq/28353318_100/'
#mp='/checkpoint/gizacard/qacache/retriever_nq/28506165_75/'
#mp='checkpoint/retriever/multiset/bert-base-encoder.cp'
#mp='/checkpoint/gizacard/qacache/retriever_nq/28750188_100'
#mp='/checkpoint/gizacard/qacache/retriever_nq/28796467_100'
#mp='/checkpoint/gizacard/qacache/retriever_nq/28931354_100'
#mp='/checkpoint/gizacard/qacache/retriever_nq/28958030_100/'

#mp='/checkpoint/gizacard/qacache/retriever_nq/28980342_100/'
mp='/checkpoint/gizacard/qacache/retriever_nq/28997530_100/'
mp='/checkpoint/gizacard/qacache/retriever_nq/29009550_100/'
mp='/checkpoint/gizacard/qacache/retriever_nq/29030159_100/checkpoint/step-38500' #2iter ranking loss w finetuning
mp='/checkpoint/gizacard/qacache/retriever_nq/29030049_200/checkpoint/best_dev/'
mp='/checkpoint/gizacard/qacache/retriever_nq/29029673_200/checkpoint/best_dev/'
mp='/checkpoint/gizacard/qacache/retriever_nq/29046892_100/checkpoint/step-20000' #KL i1
#mp='/checkpoint/gizacard/qacache/retriever_nq/29070923_100/checkpoint/step-42500' #KL trainin i2
#mp='/checkpoint/gizacard/qacache/retriever_nq/29055365_100/checkpoint/step-20000'
#mp='/checkpoint/gizacard/qacache/retriever_nq/29055727_100/checkpoint/step-20000' #KL training 0.8
#mp='/checkpoint/gizacard/qacache/retriever_nq/29115828_100/checkpoint/step-58000' #KL training i3
#mp='/checkpoint/gizacard/qacache/retriever_nq/29158690_200/checkpoint/step-22000' #100 dpr + 50 gold +50 random
#mp='/checkpoint/gizacard/qacache/retriever_nq/29326592_100/checkpoint/step-22500' #dropout 01
#mp='/checkpoint/gizacard/qacache/retriever_nq/29322506_100/checkpoint/step-23500' #dropout 05
#mp='/checkpoint/gizacard/qacache/retriever_nq/29356339_100/checkpoint/step-32000' #KL i1 Trivia
#mp='/checkpoint/gizacard/qacache/retriever_nq/29356341_100/checkpoint/step-32000' #KL i1 Trivia roberta
#mp='/checkpoint/gizacard/qacache/retriever_nq/29510777_100/checkpoint/step-43500' #KL i2 Trivia
mp='/checkpoint/gizacard/qacache/retriever_nq/29717989_100_154/checkpoint/step-25000/' #KL i1 5 x 154
mp='/checkpoint/gizacard/qacache/retriever_nq/29618650_100_1400/checkpoint/step-30000/' #KL i1 5 x 768
mp='/checkpoint/gizacard/qacache/retriever_nq/30336868_100/checkpoint/step-20000' #i1 bm 25
mp='/checkpoint/gizacard/qacache/retriever_nq/30357826_100/checkpoint/step-60000/' #i3 bert init nq 


split=dev

~/anaconda3/envs/dpr/bin/python3 dense_retriever.py \
    --model_file $mp \
    --ctx_file data/wikipedia_split/psgs_w100.tsv \
    --qa_file data/retriever/qas/nq-$split.csv \
    --encoded_ctx_file "data/nq_bertinit_i3/wiki_passages_*" \
    --out_file contexts/nq_bertinit_i3/dpr_nq_$split.json \
    --n-docs 100 \
    --validation_workers 64 \
    --batch_size 128 \
    --encoder_model_type hf_attention \
    --index_buffer 500000 \

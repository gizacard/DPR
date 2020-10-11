#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:40:00
#SBATCH --job-name=gdpr
#SBATCH --output=/private/home/%u/DPR/run_dir/%A
#SBATCH --partition=priority
#SBATCH --mem=20GB
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --array=0-15
#SBATCH --comment="iclr"

mp='/checkpoint/gizacard/qacache/retriever_nq/29115828_100/checkpoint/step-58000'
mp='/checkpoint/gizacard/qacache/retriever_nq/30048689_100_16/checkpoint/step-10000/'
mp=/checkpoint/gizacard/qacache/retriever_nq/30846322_100/checkpoint/step-15000
mp='checkpoint/retriever/multiset/bert-base-encoder.cp'
mp='/checkpoint/gizacard/qacache/retriever_nq/30977515_100/checkpoint/step-30000/'
split=test
counter=0
search_dir='/private/home/gizacard/fid/data/narrativeqa/'$split
doc_dir='/private/home/gizacard/fid/data/narrativeqa/preprocessed_documents'
out_dir=contexts/narrativeqa_bm25_i2/$split
mkdir -p $out_dir
for entry in $search_dir/*
do
  modint=$(( $((counter++)) % 16 ))
  if (( $modint == $SLURM_ARRAY_TASK_ID ));
    then
    doc_id=$(basename "$entry" .tsv)
    #doc_path="$search_dir"/"$doc_id".json
    out_path=$out_dir/"$doc_id".json
    doc_path="$doc_dir"/"$doc_id".tsv
    qa_path="$search_dir"/"$doc_id".tsv
    enc_path=data/narrativeqa_bm25_i2/"$doc_id"_0
    #mkdir -p $enc_path
    ~/anaconda3/envs/dpr/bin/python3 dense_retriever.py --model_file $mp --ctx_file $doc_path --qa_file $qa_path --encoded_ctx_file $enc_path --out_file $out_path --n-docs 100 --validation_workers 64 --batch_size 128 --encoder_model_type hf_attention --index_buffer 200000
  fi
done


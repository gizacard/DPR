#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:40:00
#SBATCH --job-name=gdpr
#SBATCH --output=/private/home/%u/DPR/run_dir/%A
#SBATCH --partition=dev
#SBATCH --mem=20GB
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --array=0-15
#SBATCH --comment="iclr"

#mp='/checkpoint/gizacard/qacache/retriever_nq/29115828_100/checkpoint/step-58000'
#mp='/checkpoint/gizacard/qacache/retriever_nq/30048689_100_16/checkpoint/step-10000/'
mp='checkpoint/retriever/multiset/bert-base-encoder.cp'
#mp=/checkpoint/gizacard/qacache/retriever_nq/30846322_100/checkpoint/step-15000
mp='/checkpoint/gizacard/qacache/retriever_nq/30977515_100/checkpoint/step-30000/'
counter=0
search_dir='/private/home/gizacard/fid/data/narrativeqa/preprocessed_documents'
for entry in $search_dir/*
do
  modint=$(( $((counter++)) % 16 ))
  if (( $modint == $SLURM_ARRAY_TASK_ID ));
    then
    doc_id=$(basename "$entry" .tsv)
    doc_path="$search_dir"/"$doc_id".tsv
    enc_path=data/narrativeqa_bm25_i2/"$doc_id"
    #mkdir -p $enc_path

~/anaconda3/envs/dpr/bin/python3 generate_dense_embeddings.py --model_file $mp --ctx_file $doc_path --shard_id 0 --num_shards 1 --out_file $enc_path --batch_size 256 --encoder_model_type hf_attention

  fi
done


mp=none
mp=/checkpoint/gizacard/qacache/retriever_nq/31232096_100/checkpoint/best_dev/ #fever bm25 i1

split=train

python dense_retriever.py \
    --save_or_load_index \
    --model_file $mp \
    --ctx_file data/wikipedia_split/psgs_w100.tsv \
    --qa_file data/retriever/qas/fever-$split.csv \
    --encoded_ctx_file "data/fever_bm25_i1/wiki_passages_*" \
    --out_file contexts/fever_bm25_i1/$split.json \
    --n-docs 100 \
    --validation_workers 64 \
    --batch_size 128 \
    --encoder_model_type hf_attention \
    --index_buffer 1000000 \
    --index_type flat \

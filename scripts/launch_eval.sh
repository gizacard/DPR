mp='/checkpoint/gizacard/qacache/retriever_nq/29902766_100_16/checkpoint/step-35000' #16 dim colbert
split=test

python evaluate.py \
    --save_or_load_index \
    --model_file $mp \
    --ctx_file data/wikipedia_split/psgs_w100.tsv \
    --qa_file data/retriever/qas/nq-$split.csv \
    --encoded_ctx_file "data/wi_colbert16/wiki_passages_*" \
    --out_file contexts/nq_colbert16/$split.json \
    --n-docs 100 \
    --validation_workers 2 \
    --batch_size 128 \
    --encoder_model_type colbert \
    --index_buffer 1000000 \
    --index_type custom \

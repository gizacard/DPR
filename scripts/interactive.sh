args=w100
mp='/checkpoint/gizacard/qacache/retriever_nq/28353318_100/'
#mp='/checkpoint/gizacard/qacache/retriever_nq/29717989_100_154/checkpoint/step-25000/'
#mp='/checkpoint/gizacard/qacache/retriever_nq/29882021_100_154/checkpoint/best_dev/'
mp='/checkpoint/gizacard/qacache/retriever_nq/29902766_100_16/checkpoint/step-35000'
#mp='/checkpoint/gizacard/qacache/retriever_nq/30048689_100_16/checkpoint/step-10000/'
#mp='checkpoint/retriever/multiset/bert-base-encoder.bin'
ty=hf_attention
mp=none

#mp=checkpoint/retriever/multiset/bert-base-encoder.cp 

~/anaconda3/envs/dpr/bin/python3 generate_dense_embeddings.py \
    --model_file $mp \
    --ctx_file data/wikipedia_split/psgs_$args.tsv \
    --shard_id 0 \
    --num_shards 50 \
    --out_file data/interactive/wiki_passages \
    --batch_size 2 \
    --encoder_model_type $ty \

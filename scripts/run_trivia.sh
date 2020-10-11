mp=none
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30340124_100/checkpoint/step-15000/' #i1 bert ini trivia
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30336886_100/checkpoint/step-20000/' #i1 bm25 trivia
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30457958_100/checkpoint/step-40000' #i2 bertinit trivia
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30465688_100/checkpoint/step-45000' #trivia i2 bm25
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30540366_100/checkpoint/step-35000' #i2 trivia bm25
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30642235_100/checkpoint/step-75000' #i4 trivia bertinit
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30681162_100/checkpoint/step-50000' #i3 trivia bm25
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30746162_100/checkpoint/step-85000/' #i5 trivia bertinit
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30746162_100/checkpoint/step-90000/' #i5 trivia bertinit 90k
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30771650_100/checkpoint/step-95000' #i6 trivia bertinit 
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30788158_100/checkpoint/step-100000/' #i7 trivia bertinit
mp=/checkpoint/gizacard/qacache/retriever_trivia/31062828_100/checkpoint/step-15000/ #trivia bert init i1 __ second run
mp=/checkpoint/gizacard/qacache/retriever_trivia/31101965_100/checkpoint/step-30000/ #trivia bert init i2 _ second run
mp=/checkpoint/gizacard/qacache/retriever_trivia/31167445_100/checkpoint/step-45000/ #trivia bert init i3 _ second run
mp=/checkpoint/gizacard/qacache/retriever_trivia/31215820_100/checkpoint/step-55000/ #trivia bert init i4 _ second run
mp=/checkpoint/gizacard/qacache/retriever_trivia/31246295_100/checkpoint/step-65000/ #trivia bert init i5 _ second run

split=test

python dense_retriever.py \
    --save_or_load_index \
    --model_file $mp \
    --ctx_file data/wikipedia_split/psgs_w100.tsv \
    --qa_file data/retriever/qas/trivia-$split.csv \
    --encoded_ctx_file "data/_trivia_bertinit_i5/wiki_passages_*" \
    --out_file contexts/_trivia_bertinit_i5/$split.json \
    --n-docs 100 \
    --validation_workers 64 \
    --batch_size 128 \
    --encoder_model_type hf_attention \
    --index_buffer 1000000 \
    --index_type flat \

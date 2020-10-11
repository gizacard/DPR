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
#mp='/checkpoint/gizacard/qacache/retriever_nq/29717989_100_154/checkpoint/step-25000/' #KL i1 5 x 154
#mp='/checkpoint/gizacard/qacache/retriever_nq/29618650_100_1400/checkpoint/step-30000/' #KL i1 5 x 768
#mp='/checkpoint/gizacard/qacache/retriever_nq/29882021_100_154/checkpoint/best_dev/'
#mp='/checkpoint/gizacard/qacache/retriever_nq/29902766_100_16/checkpoint/step-35000' #16 dim colbert
#mp='/checkpoint/gizacard/qacache/retriever_nq/30119395_100/checkpoint/step-25000/' #cadd 1.
mp='/checkpoint/gizacard/qacache/retriever_nq/30120387_100/checkpoint/step-25000/' #cadd 0. csub 0.
#mp='/checkpoint/gizacard/qacache/retriever_nq/30120438_100/checkpoint/step-25000/' #csub 1.
#mp='/checkpoint/gizacard/qacache/retriever_nq/30179740_100/checkpoint/step-25000' #cadd 2.
#mp='/checkpoint/gizacard/qacache/retriever_nq/30203910_40/checkpoint/step-25000' #40 passages training
#mp=none


split=train

python dense_retriever.py \
    --save_or_load_index \
    --model_file $mp \
    --ctx_file data/wikipedia_split/psgs_w100.tsv \
    --qa_file data/retriever/qas/eli5-$split.csv \
    --encoded_ctx_file "data/wi_i1_cadd0_csub0/wiki_passages_*" \
    --out_file contexts/wi_i1_cadd0_csub0/$split.json \
    --n-docs 100 \
    --validation_workers 64 \
    --batch_size 128 \
    --encoder_model_type hf_attention \
    --index_buffer 1000000 \
    --index_type flat \

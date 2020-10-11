dir=contexts/narrativeqa_bm25/
#dir=/private/home/gizacard/fid/preprocessed_data/nq/nq_bm25_
split=dev
dir=contexts/nq_mm_i1_1_20k/
#dir=contexts/wi_bertinit_i1/
#dir=contexts/trivia_bertinit_i6/
#dir=contexts/trivia_bm25_i3/
#dir=/private/home/gizacard/fid/preprocessed_data/trivia/trivia_bm25_
#dir=~/DPR/contexts/nq_mean_i3_kl/dpr_nq_
dir=contexts/nq_bm25maxlayers_i0/


python evaluate_topk.py \
    --output_file "$dir""$split".json \
    --n-docs 100 \
    --validation_workers 16 \
    --batch_size 128 \

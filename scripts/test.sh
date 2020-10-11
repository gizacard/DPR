mp='/checkpoint/gizacard/qacache/retriever_nq/29115828_100/checkpoint/step-58000'
#mp='/checkpoint/gizacard/qacache/retriever_nq/30048689_100_16/checkpoint/step-10000/'
mp='checkpoint/retriever/multiset/bert-base-encoder.cp'
split=dev
counter=0
search_dir='/private/home/gizacard/fid/data/narrativeqa/'$split
doc_dir='/private/home/gizacard/fid/data/narrativeqa/preprocessed_documents'
out_dir=contexts/narrativeqa_dpr/$split
mkdir -p $out_dir
for entry in $search_dir/*
do
    echo $entry
    doc_id=$(basename "$entry" .tsv)
    #doc_path="$search_dir"/"$doc_id".json
    out_path=$out_dir/"$doc_id".json
    doc_path="$doc_dir"/"$doc_id".tsv
    qa_path="$search_dir"/"$doc_id".tsv
    enc_path=data/narrativeqa_dpr/"$doc_id"_0
    #mkdir -p $enc_path
    ~/anaconda3/envs/dpr/bin/python3 dense_retriever.py --model_file $mp --ctx_file $doc_path --qa_file $qa_path --encoded_ctx_file $enc_path --out_file $out_path --n-docs 100 --validation_workers 64 --batch_size 128 --encoder_model_type hf_bert --index_buffer 200000
done


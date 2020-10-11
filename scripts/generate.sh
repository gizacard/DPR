#!/bin/bash
#SBATCH --cpus-per-task=3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=1:40:00
#SBATCH --job-name=gdpr
#SBATCH --output=/private/home/%u/DPR/run_dir/%A
#SBATCH --partition=priority
#SBATCH --mem=60GB
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --array=0-49
#SBATCH --comment="eacl"

#wikipath = data/wikipedia_split/psgs_w100.tsv
#wikipath = '/private/home/gizacard/qacache/wiki_file.tsv'
args=w100
#mp='/checkpoint/gizacard/qacache/retriever_nq/28353318_100/'
#mp='/checkpoint/gizacard/qacache/retriever_nq/28506165_75/'
#mp='/checkpoint/gizacard/qacache/retriever_nq/28750188_100'
#mp='/checkpoint/gizacard/qacache/retriever_nq/28796467_100'
#mp='checkpoint/retriever/multiset/bert-base-encoder.cp' 
#mp='/checkpoint/gizacard/qacache/retriever_nq/28931354_100'
#mp='/checkpoint/gizacard/qacache/retriever_nq/28958030_100/'
#mp='/checkpoint/gizacard/qacache/retriever_nq/28980342_100/'
mp='/checkpoint/gizacard/qacache/retriever_nq/28997530_100/'
mp='/checkpoint/gizacard/qacache/retriever_nq/29009550_100/'
mp='/checkpoint/gizacard/qacache/retriever_nq/29030159_100/checkpoint/step-38500' #2iter ranking loss w finetuning
mp='/checkpoint/gizacard/qacache/retriever_nq/29030049_200/checkpoint/best_dev/' #additional passages 50 wikies 50 golds
mp='/checkpoint/gizacard/qacache/retriever_nq/29029673_200/checkpoint/best_dev/' #adaptive loss additional passage 50 wikis 50 golds
#mp='/checkpoint/gizacard/qacache/retriever_nq/29046892_100/checkpoint/step-20000' #KL training
mp='/checkpoint/gizacard/qacache/retriever_nq/29055727_100/checkpoint/step-20000' #KL training 0.8
#mp='/checkpoint/gizacard/qacache/retriever_nq/29055365_100/checkpoint/step-20000' #KL training 1.5
mp='/checkpoint/gizacard/qacache/retriever_nq/29070923_100/checkpoint/step-42500' #KL trainin i2
mp='/checkpoint/gizacard/qacache/retriever_nq/29115828_100/checkpoint/step-58000' #KL training i3
mp='/checkpoint/gizacard/qacache/retriever_nq/29326592_100/checkpoint/step-22500' #dropout 01
mp='/checkpoint/gizacard/qacache/retriever_nq/29322506_100/checkpoint/step-23500' #dropout 05
mp='/checkpoint/gizacard/qacache/retriever_nq/29356339_100/checkpoint/step-32000' #KL i1 Trivia
mp='/checkpoint/gizacard/qacache/retriever_nq/29356341_100/checkpoint/step-32000' #KL i1 Trivia roberta
mp='/checkpoint/gizacard/qacache/retriever_nq/29510777_100/checkpoint/step-43500' #KL i2 Trivia
mp='/checkpoint/gizacard/qacache/retriever_nq/29717989_100_154/checkpoint/step-25000/' #KL i1 5 x 154 
mp='/checkpoint/gizacard/qacache/retriever_nq/29618650_100_1400/checkpoint/step-30000/' #KL i1 5 x 768
mp='/checkpoint/gizacard/qacache/retriever_nq/29882021_100_154/checkpoint/best_dev/'
mp='/checkpoint/gizacard/qacache/retriever_nq/29902766_100_16/checkpoint/step-35000' #16dim colbert
#mp='/checkpoint/gizacard/qacache/retriever_nq/30119395_100/checkpoint/step-25000/' #cadd 1.
#mp='/checkpoint/gizacard/qacache/retriever_nq/30120387_100/checkpoint/step-25000/' #cadd 0. csub 0.
#mp='/checkpoint/gizacard/qacache/retriever_nq/30120438_100/checkpoint/step-25000/' #csub 1.
#mp='/checkpoint/gizacard/qacache/retriever_nq/30179740_100/checkpoint/step-25000' #cadd 2.
#mp='/checkpoint/gizacard/qacache/retriever_nq/30203910_40/checkpoint/step-25000' #40 passages training
#mp='/checkpoint/gizacard/qacache/retriever_nq/30247597_100/checkpoint/step-15000' #i1 bert init nq
#mp='/checkpoint/gizacard/qacache/retriever_nq/30291112_100/checkpoint/step-45000' #i2 bert init nq
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30340124_100/checkpoint/step-15000/' #i1 bert ini trivia
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30336886_100/checkpoint/step-20000/' #i1 bm25 trivia
#mp='/checkpoint/gizacard/qacache/retriever_nq/30336868_100/checkpoint/step-20000' #i1 bm25 nq
#mp='/checkpoint/gizacard/qacache/retriever_nq/30357826_100/checkpoint/step-60000/' #i3 bert init nq
#mp='/checkpoint/gizacard/qacache/retriever_nq/30453185_100/checkpoint/step-80000/' #i4 bertinit nq
#mp='/checkpoint/gizacard/qacache/retriever_nq/30420904_100/checkpoint/step-40000' #i2 bm 25 nq
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30457958_100/checkpoint/step-40000' #i2 bertinit trivia
#mp='/checkpoint/gizacard/qacache/retriever_nq/30483490_100/checkpoint/step-55000/' #i3 bm25 nq
#mp='/checkpoint/gizacard/qacache/retriever_nq/30484525_100/checkpoint/step-25000/' #i4 bertinit nq
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30514848_100/checkpoint/step-60000' #i3 trivia bertinit
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30540366_100/checkpoint/step-35000' #i2 trivia bm25
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30642235_100/checkpoint/step-75000' #i4 trivia bertinit
#mp='/checkpoint/gizacard/qacache/retriever_nq/30644092_100/checkpoint/step-20000/' #ablation l2 nq
#mp='/checkpoint/gizacard/qacache/retriever_nq/30644231_100/checkpoint/step-25000/' #ablation mm nq 1.0
#mp='/checkpoint/gizacard/qacache/retriever_nq/30672212_100/checkpoint/step-20000/' #ablation mm nq 0.2
#mp='/checkpoint/gizacard/qacache/retriever_nq/30677218_100/checkpoint/step-20000/' #ablation mm nq 0.1
#mp='/checkpoint/gizacard/qacache/retriever_nq/30644231_100/checkpoint/step-20000/' #ablation mm nq 1.0 20k
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30681162_100/checkpoint/step-50000' #i3 trivia bm25
#mp='/checkpoint/gizacard/qacache/retriever_nq/30725366_100/checkpoint/step-20000' #i1 bm25 nq max
#mp='/checkpoint/gizacard/qacache/retriever_nq/30725695_100/checkpoint/step-20000' #i1 bm25 nq mean
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30746162_100/checkpoint/step-85000/' #i5 trivia bertinit
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30771650_100/checkpoint/step-95000' #i6 trivia bertinit
#mp='/checkpoint/gizacard/qacache/retriever_trivia/30788158_100/checkpoint/step-100000/' #i7 trivia bertinit
#mp='/checkpoint/gizacard/qacache/retriever_nq/30819281_100/checkpoint/step-20000' #ablation last6
#mp='/checkpoint/gizacard/qacache/retriever_nq/30845056_100/checkpoint/step-20000' #nq maxheads i0
#mp='/checkpoint/gizacard/qacache/retriever_nq/30845059_100/checkpoint/step-20000' #nq maxlayers i0
#mp=/checkpoint/gizacard/qacache/retriever_trivia/31062828_100/checkpoint/step-15000/ #trivia bert init i1 _ second run
#mp=/checkpoint/gizacard/qacache/retriever_trivia/31101965_100/checkpoint/step-30000/ #trivia bert init i2 _ second run
#mp=/checkpoint/gizacard/qacache/retriever_nq/31099929_100/checkpoint/step-20000/ #trivia bm25 i0 5k
#mp=/checkpoint/gizacard/qacache/retriever_nq/31099927_100/checkpoint/step-20000/ #trivia bm25 i0 10k
#mp=/checkpoint/gizacard/qacache/retriever_nq/31099921_100/checkpoint/step-20000/ #trivia bm25 i0 15k
#mp=/checkpoint/gizacard/qacache/retriever_trivia/31167445_100/checkpoint/step-45000/ #trivia bert init i3 _ second run
#mp=/checkpoint/gizacard/qacache/retriever_trivia/31215820_100/checkpoint/step-55000/ #trivia bert init i4 _ second run
#mp=/checkpoint/gizacard/qacache/retriever_nq/31232096_100/checkpoint/best_dev/ #fever bm25 i1
#mp=/checkpoint/gizacard/qacache/retriever_trivia/31246295_100/checkpoint/step-65000 #trivia bert init i5 _ second run



~/anaconda3/envs/dpr/bin/python3 generate_dense_embeddings.py \
    --model_file $mp \
    --ctx_file data/wikipedia_split/psgs_$args.tsv \
    --shard_id $SLURM_ARRAY_TASK_ID \
    --num_shards 50 \
    --out_file data/wi_colbert16/wiki_passages \
    --batch_size 256 \
    --encoder_model_type colbert \

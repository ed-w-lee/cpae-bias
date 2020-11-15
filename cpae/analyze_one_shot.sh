#!/bin/sh

export PYTHONPATH=$PWD:$PWD/word_embeddings_benchmarks:$PYTHONPATH
export FUEL_DATA_PATH=$PWD"/data/en_wn_split"
# use the path of the embedding file of the model selected by the model selection procedure.
# EMBEDDING_FN="../embeddings/cpae-lam8.pkl"
# EMBEDDING_FN="embeddings/en_wn_full/s2sg_c300_pen8_c1_F/if_mis_e_embeddings.pkl"
EMBEDDING_FN="embeddings/en_wn_pronouns/s2sg_c300_pen8_c1_F/if_mis_e_embeddings.pkl"
EMBEDDING_NAME="cpae_pronouns"
DATA_DIR="data/en_wn_split/"
RESULTS_DIR="results/one_shot_analysis"

mkdir -p $RESULTS_DIR

echo $PYTHONPATH

python bin/eval_embs.py $EMBEDDING_FN dict_poly $DATA_DIR > $RESULTS_DIR/$EMBEDDING_NAME

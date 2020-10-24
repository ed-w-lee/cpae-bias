# Embeddings

This contains the list of embeddings tested.
They're quite large so they've been added to the `.gitignore`.
Some of these were quite annoying to process, so if anyone happens to be reading and wants these embeddings, just email me if I have an email public or something.

## List of Embeddings

Embedding | Pre-trained | Expected Name | Source
--|--|--|--
**CPAE** (one-shot, lambda=8) | No | `cpae-lam8.pkl` | [paper](https://www.aclweb.org/anthology/D18-1181.pdf) and [source](https://github.com/tombosc/cpae)
GoogleNews word2vec (slimmed down by eyaler) | Yes | `google-news-w2v.bin` | [slimmed](https://github.com/eyaler/word2vec-slim) and [original](https://code.google.com/archive/p/word2vec/)
GLoVe (Wikipedia 2014 + Gigaword 5) | Yes | `glove.6B.300d.txt` | https://nlp.stanford.edu/projects/glove/
dict2vec | Yes | `dict2vec-300d.vec` | https://github.com/tca19/dict2vec
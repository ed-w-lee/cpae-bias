# CPAE Bias Measurement

This is a project for [CS230](cs230.stanford.edu) aimed at measuring the amount of bias present in a dictionary-derived formulation of word embeddings.
More specifically, this attempts to measure the gender bias present in Tom Bosc and Pascal Vincent's [CPAE](https://www.aclweb.org/anthology/D18-1181.pdf) and see how it compares to other word embeddings derived from less structured corpora.

## TODO
[x] Train CPAE embeddings
[x] Gather baseline embeddings (e.g. Google News Word2Vec and GloVe)
[ ] Evaluate bias using WEAT tests
[ ] Train CPAE-P embeddings using w2v or something
### Tentative
[ ] Try to examine CPAE and/or CPAE-P at a deeper level (e.g. can we do PCA or something?)
[ ] Find new formulation of dict2vec or CPAE maybe?
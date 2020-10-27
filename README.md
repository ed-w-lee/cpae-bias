# CPAE Bias Measurement

This is a project for [CS230](cs230.stanford.edu) aimed at measuring the amount of bias present in a dictionary-derived formulation of word embeddings.
More specifically, this attempts to measure the gender bias present in Tom Bosc and Pascal Vincent's [CPAE](https://www.aclweb.org/anthology/D18-1181.pdf) and see how it compares to other word embeddings derived from less structured corpora.

## For milestone
- [x] Train CPAE embeddings
- [x] Gather baseline embeddings (e.g. Google News Word2Vec and GloVe)
- [x] Evaluate bias using WEAT tests
- [x] Try to examine CPAE at a deeper level (e.g. can we do PCA or something?)
- [ ] Look at dictionaries being used by dict2vec and CPAE -- can we see where connections go for each word? try to form an understanding of why we still see bias when formulating these definitions for dictionaries
### Tentative
- [ ] Try removing "offending" words (e.g. proper nouns) to see if that might affect something
- [ ] Find other possible sources of bias, or a different construction of word embeddings that may not be affected by this as much.
- [ ] Find new formulation of dict2vec or CPAE maybe?

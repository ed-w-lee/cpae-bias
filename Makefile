.PHONY: setup init clean notebook

setup: clean init

init:
	python eval/embedding.py

clean:
	rm -f embeddings/*.cache embeddings/*.filter.bin

notebook:
	PYTHONPATH='$PYTHONPATH:/home/edwlee/Documents/stanford/cs230/cpae-bias' jupyter notebook

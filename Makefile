.PHONY: setup init clean

setup: clean init

init:
	python eval/embedding.py

clean:
	rm -f embeddings/*.cache embeddings/*.filter.bin

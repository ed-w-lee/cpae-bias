import numpy as np
import os
import pickle

from gensim import models


def validate_embeds(embeds):
    '''
    Validates that a set of embeddings is of the expected shape. 
    Namely, they look like:
    { lowercase word : numpy array with shape=(300,) dtype=float32 }
    '''
    return isinstance(embeds, dict) \
        and all(k.islower() for k in embeds.keys()) \
        and all(isinstance(v, np.ndarray)
                and v.shape == (300,)
                and v.dtype == np.float32 for v in embeds.values())


class Embedding:
    '''
    Base class for word embeddings. I thought about using gensim's model, but
    I don't really want to read through the docs tbh.
    '''

    def __init__(self, parse_func, cache_path=None):
        if os.path.exists(cache_path):
            # attempt to read from cache
            with open(cache_path, 'rb') as f:
                self.embeds = pickle.load(f, encoding='bytes')
            return
        # no cache, parse as needed and save to cache
        self.embeds = parse_func()

        if cache_path:
            with open(cache_path, 'wb+') as f:
                pickle.dump(self.embeds, f)


class EmbedCpae(Embedding):
    def __init__(
        self,
        embed_path='embeddings/cpae-lam8.pkl',
        cache_path='embeddings/cpae-lam8.pkl.cache',
    ):
        def read_embeds():
            with open(embed_path, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                return {k: v[0] for k, v in data.items()}

        super().__init__(read_embeds, cache_path=cache_path)


class EmbedGlove(Embedding):
    def __init__(
        self,
        embed_path='embeddings/glove.6B.300d.txt',
        cache_path='embeddings/glove.6B.300d.pkl.cache',
        words=None,
    ):
        def read_embeds():
            with open(embed_path, 'r') as f:
                data = {}
                for l in f:
                    toks = l.split()
                    assert(len(toks) == 301)
                    word, embed = (toks[0], [float(e) for e in toks[1:]])
                    if words and word in words:
                        data[word] = np.array(embed, dtype=np.float32)
            return data

        super().__init__(read_embeds, cache_path=cache_path)


class EmbedWord2V(Embedding):
    def __init__(
        self,
        embed_path='embeddings/google-news-w2v.bin',
        cache_path='embeddings/google-news-w2v.pkl.cache',
        words=None,
    ):
        def read_embeds():
            gs = models.KeyedVectors.load_word2vec_format(
                embed_path, binary=True)
            data = {}
            for word in gs.vocab:
                if words and word in words:
                    data[word] = np.array(gs[word], dtype=np.float32)
            return data

        super().__init__(read_embeds, cache_path=cache_path)


class EmbedDict2V(Embedding):
    def __init__(
        self,
        embed_path='embeddings/dict2vec-300d.vec',
        cache_path='embeddings/dict2vec-300d.pkl.cache',
        words=None
    ):
        def read_embeds():
            with open(embed_path, 'r') as f:
                data = {}
                f.readline()  # discard first line
                for l in f:
                    toks = l.split()
                    assert(len(toks) == 301)
                    word, embed = (toks[0], [float(e) for e in toks[1:]])
                    if words and word in words:
                        data[word] = np.array(embed, dtype=np.float32)
            return data

        super().__init__(read_embeds, cache_path=cache_path)


if __name__ == "__main__":
    def parse_and_validate(name, init):
        print(f'Parsing {name} embeddings...')
        model = init()
        print(f'-> Validating {name} embeddings...')
        validate_embeds(model.embeds)
        print(f'---> Successfully parsed and validated {name} embeddings.')
        print(f'     Found {len(model.embeds)} words.')
        return model
    cpae = parse_and_validate('CPAE', EmbedCpae)
    cpae_words = set(cpae.embeds.keys())

    parse_and_validate('GloVe', lambda: EmbedGlove(words=cpae_words))
    parse_and_validate('Word2V', lambda: EmbedWord2V(words=cpae_words))
    parse_and_validate('Dict2V', lambda: EmbedDict2V(words=cpae_words))

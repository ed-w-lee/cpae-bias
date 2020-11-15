import gensim
import numpy as np
import os
import pickle

from gensim import models
from tqdm import tqdm
from pathlib import Path

parent_dir = Path(os.path.realpath(__file__)).parent.parent


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

    def __init__(self, parse_func, embed_path, cache_path, gs_path):
        self._parse_func = parse_func
        self.embed_path = embed_path
        self.cache_path = cache_path
        self.gs_path = gs_path

        self.np_model = None
        self.gs_model = None

    def get_np(self):
        if self.np_model:
            return self.np_model

        if os.path.exists(self.cache_path):
            # attempt to read from cache
            with open(self.cache_path, 'rb') as f:
                embeds = pickle.load(f, encoding='bytes')
        else:
            # no cache, parse as needed and save to cache
            embeds = self._parse_func()

            if self.cache_path:
                with open(self.cache_path, 'wb+') as f:
                    pickle.dump(embeds, f)

            # write gensim-parseable path, if needed
            if self.gs_path:
                with gensim.utils.open(self.gs_path, 'wb+') as fout:
                    fout.write(gensim.utils.to_utf8(
                        f'{len(embeds)} 300\n'))
                    for word, row in embeds.items():
                        fout.write(gensim.utils.to_utf8(
                            word) + b' ' + row.tostring())

        self.np_model = embeds
        return self.np_model

    def get_gs(self):
        if self.gs_model:
            return self.gs_model

        self.gs_model = models.KeyedVectors.load_word2vec_format(
            self.gs_path, binary=True)
        return self.gs_model


class EmbedCpae(Embedding):
    def __init__(
        self,
        embed_path=f'{parent_dir}/embeddings/cpae-lam8.pkl',
        cache_path=f'{parent_dir}/embeddings/cpae-lam8.pkl.cache',
        gs_path=f'{parent_dir}/embeddings/cpae-lam8.filter.bin',
    ):
        def read_embeds():
            with open(embed_path, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                return {k: v[0] for k, v in data.items()}

        super().__init__(read_embeds, embed_path, cache_path=cache_path, gs_path=gs_path)


class EmbedCpaeV2(Embedding):
    def __init__(
        self,
        embed_path=f'{parent_dir}/embeddings/cpae-noproper.pkl',
        cache_path=f'{parent_dir}/embeddings/cpae-noproper.pkl.cache',
        gs_path=f'{parent_dir}/embeddings/cpae-noproper.filter.bin',
        words=None
    ):
        def read_embeds():
            with open(embed_path, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                return {k: v[0] for k, v in data.items() if not words or k in words}

        super().__init__(read_embeds, embed_path, cache_path=cache_path, gs_path=gs_path)


class EmbedCpaeV3(Embedding):
    def __init__(
        self,
        embed_path=f'{parent_dir}/embeddings/cpae-pronoun.pkl',
        cache_path=f'{parent_dir}/embeddings/cpae-pronoun.pkl.cache',
        gs_path=f'{parent_dir}/embeddings/cpae-pronoun.filter.bin',
        words=None
    ):
        def read_embeds():
            with open(embed_path, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                return {k: v[0] for k, v in data.items() if not words or k in words}

        super().__init__(read_embeds, embed_path, cache_path=cache_path, gs_path=gs_path)


class EmbedGlove(Embedding):
    def __init__(
        self,
        embed_path=f'{parent_dir}/embeddings/glove.6B.300d.txt',
        cache_path=f'{parent_dir}/embeddings/glove.6B.300d.pkl.cache',
        gs_path=f'{parent_dir}/embeddings/glove.6B.300d.filter.bin',
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

        super().__init__(read_embeds, embed_path, cache_path=cache_path, gs_path=gs_path)


class EmbedWord2V(Embedding):
    def __init__(
        self,
        embed_path=f'{parent_dir}/embeddings/google-news-w2v.bin',
        cache_path=f'{parent_dir}/embeddings/google-news-w2v.pkl.cache',
        gs_path=f'{parent_dir}/embeddings/google-news-w2v.filter.bin',
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

        super().__init__(read_embeds, embed_path, cache_path=cache_path, gs_path=gs_path)


class EmbedDict2V(Embedding):
    def __init__(
        self,
        embed_path=f'{parent_dir}/embeddings/dict2vec-300d.vec',
        cache_path=f'{parent_dir}/embeddings/dict2vec-300d.pkl.cache',
        gs_path=f'{parent_dir}/embeddings/dict2vec-300d.filter.bin',
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

        super().__init__(read_embeds, embed_path, cache_path=cache_path, gs_path=gs_path)


if __name__ == "__main__":
    print('parent_dir:', parent_dir)

    def parse_and_validate(name, init):
        print(f'Parsing {name} embeddings...')
        model = init()
        embeds = model.get_np()
        print(f'-> Validating {name} embeddings...')
        validate_embeds(embeds)
        print(f'---> Successfully parsed and validated {name} embeddings.')
        print(f'     Found {len(embeds)} words.')
        return model
    cpae = parse_and_validate('CPAE', EmbedCpae)
    cpae_words = set(cpae.get_np().keys())

    parse_and_validate('CPAEv2', lambda: EmbedCpaeV2(words=cpae_words))
    parse_and_validate('CPAEv3', lambda: EmbedCpaeV3(words=cpae_words))
    parse_and_validate('GloVe', lambda: EmbedGlove(words=cpae_words))
    parse_and_validate('Word2V', lambda: EmbedWord2V(words=cpae_words))
    parse_and_validate('Dict2V', lambda: EmbedDict2V(words=cpae_words))

import argparse
import multiprocessing
from itertools import islice
import re

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument, FAST_VERSION
import json_lines

from hh_page_clf.utils import get_stop_words


def train(input_jlgz, *, size, limit, min_df, max_features):
    print('FAST_VERSION', FAST_VERSION)
    documents = Documents(input_jlgz, limit=limit)
    model = Doc2Vec(
        documents=documents,
        size=size,
        min_count=min_df,
        max_vocab_size=max_features,
        workers=multiprocessing.cpu_count(),
        sample=1e-5,
    )
    return model


class Documents:
    def __init__(self, input_jlgz, limit=None):
        self.stop_words = get_stop_words()
        self.input_jlgz = input_jlgz
        self.limit = limit

    def __iter__(self):
        it = self._iter()
        return islice(it, self.limit) if self.limit else it

    def _iter(self):
        with json_lines.open(self.input_jlgz) as f:
            for idx, item in enumerate(f):
                tokens = tokenize(item['text'])
                if tokens:
                    yield TaggedDocument(tokens, [idx])


def tokenize(text):
    stop_words = get_stop_words()
    return [token for token in re.findall(r'(?u)\b\w\w+\b', text.lower())
            if token not in stop_words]


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('input', help='In .jl.gz format with text in "text" field')
    arg('output', help='Output doc2vec model in doc2vec format')
    arg('--size', type=int, default=300, help='embedding size')
    arg('--min-df', type=int, default=10)
    arg('--max-features', type=int, default=100000)
    arg('--limit', type=int, help='limit to first N pages')
    args = parser.parse_args()

    model = train(
        args.input,
        size=args.size,
        limit=args.limit,
        min_df=args.min_df,
        max_features=args.max_features,
    )
    model.save(args.output)

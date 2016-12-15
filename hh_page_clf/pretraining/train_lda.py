import argparse
from itertools import islice
from typing import Tuple

import json_lines
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
from tqdm import tqdm

from hh_page_clf.utils import get_stop_words


def LDAPageVctorizer(*,
                     n_topics: int,
                     min_df: int,
                     max_features: int,
                     max_iter: int,
                     ngram_range: Tuple[int, int],
                     vocabulary=None,
                     batch_size: int=4096,
                     verbose=1):
    vec = _vectorizer(min_df=min_df, max_features=max_features,
                      ngram_range=ngram_range, vocabulary=vocabulary)
    lda = LatentDirichletAllocation(
        learning_method='online',
        n_topics=n_topics,
        batch_size=batch_size,
        evaluate_every=2,
        verbose=verbose,
        max_iter=max_iter,
        n_jobs=1,
    )
    return make_pipeline(vec, lda)


def _vectorizer(*, min_df, max_features, ngram_range, vocabulary=None):
    return CountVectorizer(
        preprocessor=_text_preprocessor,
        stop_words=get_stop_words(),
        min_df=min_df,
        max_features=max_features,
        ngram_range=ngram_range,
        vocabulary=vocabulary,
    )


def _text_preprocessor(text):
    return text.lower()


def _iter_text(path):
    with json_lines.open(path) as f:
        for item in f:
            yield item['text']


def _fit_vocab(input_jlgz, *, min_df, max_features, ngram_range, limit=None,
               **_):
    max_pages = 100000
    if limit is not None and limit <= max_pages:
        return None
    data = islice(_iter_text(input_jlgz), max_pages)
    vec = _vectorizer(
        min_df=min_df, max_features=max_features, ngram_range=ngram_range)
    vec.fit(tqdm(data, desc='Fitting vectorizer'))
    return vec.vocabulary_


def train(input_jlgz, limit=None, **lda_kwargs):
    vocabulary = _fit_vocab(input_jlgz, limit=limit, **lda_kwargs)
    lda_pipe = LDAPageVctorizer(vocabulary=vocabulary, **lda_kwargs)
    data = _iter_text(input_jlgz)
    if limit:
        data = islice(data, limit)
    lda_pipe.fit(tqdm(data, desc='Loading text'))
    for name, step in lda_pipe.steps:
        step.verbose = False
    return lda_pipe


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('input', help='In .jl.gz format with text in "text" field')
    arg('output', help='Output LDA model in joblib format')
    arg('--n-topics', type=int, default=50)
    arg('--max-features', type=int, default=100000)
    arg('--ngram', type=int, default=1)
    arg('--max-iter', type=int, default=10)
    arg('--min-df', type=int, default=10)
    arg('--limit', type=int, help='limit to first N pages')
    args = parser.parse_args()

    pipe = train(
        args.input,
        n_topics=args.n_topics,
        max_features=args.max_features,
        limit=args.limit,
        ngram_range=(1, args.ngram),
        max_iter=args.max_iter,
        min_df=args.min_df,
    )
    joblib.dump(pipe, args.output, compress=3)

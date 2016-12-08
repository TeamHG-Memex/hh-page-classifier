import argparse
from itertools import islice


import json_lines
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
from tqdm import tqdm


def LDAPageVctorizer(n_topics: int, batch_size: int, min_df: int, verbose=1,
                     max_features: int=None):
    vec = CountVectorizer(
        preprocessor=_text_preprocessor,
        min_df=min_df,
        max_features=max_features,
    )
    lda = LatentDirichletAllocation(
        learning_method='online',
        n_topics=n_topics,
        batch_size=batch_size,
        evaluate_every=2,
        verbose=verbose,
        n_jobs=16,
    )
    return make_pipeline(vec, lda)


def _text_preprocessor(text):
    return text.lower()


def _iter_text(path):
    with json_lines.open(path) as f:
        for item in f:
            yield item['text']


def train(input_jlgz, n_topics=50, batch_size=1024, min_df=4,
          limit=None, max_features=None):
    lda_pipe = LDAPageVctorizer(
        n_topics=n_topics,
        batch_size=batch_size,
        min_df=min_df,
        verbose=1,
        max_features=max_features or None,
    )
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
    arg('--limit', type=int, help='limit to first N pages')
    args = parser.parse_args()

    pipe = train(
        args.input,
        n_topics=args.n_topics,
        max_features=args.max_features,
        limit=args.limit,
    )
    joblib.dump(pipe, args.output, compress=3)

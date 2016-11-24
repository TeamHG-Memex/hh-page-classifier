import argparse
import gzip
from functools import partial
try:
    import ujson as json
except ImportError:
    import json
import multiprocessing

import json_lines
import html_text
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
    )
    return make_pipeline(vec, lda)


def _text_preprocessor(text):
    return text.lower()


def _iter_text(path):
    with json_lines.open(path) as f:
        for item in f:
            yield item['text']


def train(input_jlgz, n_topics=50, batch_size=1024, min_df=4, max_features=None):
    lda_pipe = LDAPageVctorizer(
        n_topics=n_topics,
        batch_size=batch_size,
        min_df=min_df,
        verbose=1,
        max_features=max_features or None,
    )
    lda_pipe.fit(tqdm(_iter_text(input_jlgz), desc='Loading text'))
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
    args = parser.parse_args()

    pipe = train(
        args.input,
        n_topics=args.n_topics,
        max_features=args.max_features,
    )
    joblib.dump(pipe, args.output, compress=3)


def extract_texts():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('input', help='In .jl.gz format with html in "text" field')
    arg('html_field', help='Field name where html is stored')
    arg('output', help='Output in jl.gz format with text in "text" field')
    args = parser.parse_args()

    with json_lines.open(args.input) as f, gzip.open(args.output, 'wt') as outf:
        with multiprocessing.Pool() as pool:
            for text_item in pool.imap_unordered(
                    partial(text_worker, html_field=args.html_field), f):
                outf.write(json.dumps(text_item))
                outf.write('\n')


def text_worker(item, html_field):
    url = item.get('url')
    html = item.get(html_field)
    text = html_text.extract_text(html)
    text_item = {'text': text}
    if url is not None:
        text_item['url'] = url
    return text_item

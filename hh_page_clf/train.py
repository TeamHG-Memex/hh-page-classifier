import argparse
import base64
from collections import defaultdict, Counter
import gzip
import json
import logging
from multiprocessing.pool import Pool, ThreadPool
import random
from statistics import mean
import time
from typing import List, Dict

import attr
from eli5.base import FeatureWeights
from eli5.utils import max_or_0
from eli5.formatters import (
    format_as_text, format_as_dict, format_as_dataframe, fields,
)
from eli5.formatters.html import format_hsl, weight_color_hsl, get_weight_range
import html_text
import langdetect
from langdetect.lang_detect_exception import LangDetectException
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import tldextract

from . import models
from .models import BaseModel
from .utils import decode_object, encode_object, configure_logging, PBar


ERROR = 'Error'
WARNING = 'Warning'
NOTICE = 'Notice'


@attr.s
class AdviceItem:
    kind = attr.ib()
    text = attr.ib()


@attr.s
class DescriptionItem:
    heading = attr.ib()
    text = attr.ib()


@attr.s
class Meta:
    advice = attr.ib()  # type: List[AdviceItem]
    description = attr.ib(default=None)  # type: List[DescriptionItem]
    weights = attr.ib(default=None)  # type: FeatureWeights
    tooltips = attr.ib(default=None)  # type: Dict[str, str]


@attr.s
class ModelMeta:
    model = attr.ib()
    meta = attr.ib()  # type: Meta
    metrics = attr.ib(default=None)


def train_model(docs: List[Dict],
                model_cls=None,
                target_relevant_ratio=0.3,
                skip_validation=False,
                skip_eli5=False,
                skip_serialization_check=False,
                benchmark=False,
                random_pages=None,
                progress_callback=None,
                **model_kwargs) -> ModelMeta:
    """ Train and evaluate a model.
    docs is a list of dicts:
    {'url': url, 'html': html, 'relevant': True/False/None}.
    Return the model itself and a human-readable description of it's performance.
    """
    pbar = PBar(callback=progress_callback)

    if not docs:
        return no_docs_error()

    all_xs = [doc for doc in docs if doc.get('relevant') in [True, False]]
    logging.info('Got {:,} labeled pages'.format(len(all_xs)))
    if not all_xs:
        return no_labeled_error()
    n_relevant = sum(doc['relevant'] for doc in all_xs)
    if n_relevant == 0:
        return single_class_error(False)

    if random_pages:
        n_extra_non_relevant = max(
            0, n_relevant / target_relevant_ratio - len(all_xs))
        extra_non_relevant = sample_non_relevant(
            random_pages, n_extra_non_relevant)
        all_xs.extend(extra_non_relevant)
        docs.extend(extra_non_relevant)  # for proper report in get_meta
    elif n_relevant == len(all_xs):
        return single_class_error(True)
    random.shuffle(all_xs)
    all_ys = np.array([doc['relevant'] for doc in all_xs])
    advice = []

    logging.info('Extracting text')
    add_extracted_text(all_xs)
    pbar.progress(0.20)
    if model_cls is None:
        model_cls = select_model(
            [x for x in all_xs if not x.get('extra_non_relevant')])

    logging.info('Pre-loading models')
    model_cls(**model_kwargs)

    logging.info('Training and evaluating model')
    folds = build_folds(all_xs, all_ys, advice)
    model, metrics = train_and_evaluate(
        all_xs, all_ys, folds, model_cls, model_kwargs,
        skip_serialization_check=skip_serialization_check,
        skip_validation=skip_validation,
        benchmark=benchmark,
        pbar=pbar.make_child(0.75),
    )

    meta = get_meta(model, metrics, advice, docs, skip_eli5=skip_eli5)
    logging.info('Model meta:\n{}'.format(meta_repr(meta)))
    pbar.progress(0.05)

    return ModelMeta(model=model, meta=meta, metrics=metrics)


def no_docs_error():
    return ModelMeta(
        model=None,
        meta=Meta([AdviceItem(
            ERROR, 'Can not train a model, no pages given.')]))


def no_labeled_error():
    return ModelMeta(
        model=None,
        meta=Meta([AdviceItem(
            ERROR, 'Can not train a model, no labeled pages given.')]))


def single_class_error(is_positive):
    have, need = 'not ', ''
    if is_positive:
        have, need = need, have
    return ModelMeta(
        model=None,
        meta=Meta(
            [AdviceItem(
                ERROR,
                'Can not train a model, only {have}relevant pages in sample: '
                'need examples of {need}relevant pages too.'.format(
                    have=have, need=need,
                ))]))


def sample_non_relevant(random_pages_path: str, n_pages: int) -> List[Dict]:
    pages = []
    with gzip.open(random_pages_path, 'rt') as f:
        for line in f:
            if len(pages) >= n_pages:
                break
            page = json.loads(line)
            pages.append({
                'url': page['url'], 'html': page['text'],
                'relevant': False, 'extra_non_relevant': True,
            })
    return pages


def doc_is_extra_sampled(doc):
    return doc.get('extra_non_relevant')


def extract_features(doc):
    html = doc['html'] or ''
    if not doc_is_extra_sampled(doc):
        try:
            html = gzip.decompress(base64.b64decode(html)).decode('utf8')
        except Exception:
            pass  # support not compressed html too for a while - TODO - remove
    text = html_text.extract_text(html)
    try:
        lang = langdetect.detect(text)
    except LangDetectException:
        lang = None
    return {
        'text': text,
        'language': lang,
    }


def add_extracted_text(xs):
    with Pool() as pool:
        for doc, features in zip(
                xs, pool.imap(extract_features, xs, chunksize=10)):
            doc.update(features)


def select_model(xs, ngram_docs_threshold=0.2):
    lang_counts = dict(Counter(x['language'] for x in xs))
    lang_freqs = {lang: count / len(xs) for lang, count in lang_counts.items()}
    # https://en.wikipedia.org/wiki/Category:Writing_systems_without_word_boundaries
    # https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
    need_ngrams = {'my', 'zh-cn', 'zh-tw', 'ja', 'km', 'lo', 'za', 'th', 'bo'}
    need_ngrams_freq = sum(freq for lang, freq in lang_freqs.items()
                           if lang in need_ngrams)
    if need_ngrams_freq > ngram_docs_threshold:
        model = models.CharModel
    else:
        model = models.DefaultModel
    logging.info('{:.0%} documents need char ngrams, using {}'
                 .format(need_ngrams_freq, model.__name__))
    return model


def build_folds(all_xs, all_ys, advice):
    domains = [get_domain(doc['url']) for doc in all_xs]
    n_domains = len(set(domains))
    n_relevant_domains = len(
        {domain for domain, is_relevant in zip(domains, all_ys) if is_relevant})
    n_folds = 4
    if n_relevant_domains == 1:
        advice.append(AdviceItem(
            WARNING,
            'Only 1 relevant domain in data means that it\'s impossible to do '
            'cross-validation across domains, '
            'and will likely result in model over-fitting.'
        ))
        folds = KFold(n_splits=n_folds).split(all_xs)
    else:
        folds = (GroupKFold(n_splits=min(n_domains, n_folds))
                 .split(all_xs, groups=domains))

    if 1 < n_relevant_domains < WARN_N_RELEVANT_DOMAINS:
        advice.append(AdviceItem(
            WARNING,
            'Low number of relevant domains (just {}) '
            'might result in model over-fitting.'.format(n_relevant_domains)
        ))
    folds = two_class_folds(folds, all_ys)
    if not folds:
        folds = two_class_folds(KFold(n_splits=n_folds).split(all_xs), all_ys)
    if not folds:
        advice.append(AdviceItem(
            WARNING,
            'Can not do cross-validation, as there are no folds where '
            'training data has both relevant and non-relevant examples. '
            'There are too few domains or the dataset is too unbalanced.'
        ))
    return folds


def two_class_folds(folds, all_ys):
    return [fold for fold in folds if len(np.unique(all_ys[fold[0]])) > 1]


def train_and_evaluate(
        all_xs, all_ys, folds,
        model_cls, model_kwargs,
        pbar: PBar,
        skip_serialization_check=False,
        skip_validation=False,
        benchmark=False
        ):
    unit_progress = 1 / (len(folds) + 1)
    with ThreadPool(processes=len(folds)) as pool:
        metric_futures = []
        if folds and not skip_validation:
            metric_futures = [
                pool.apply_async(
                    eval_on_fold,
                    args=(fold, model_cls, model_kwargs, all_xs, all_ys),
                    kwds=dict(
                        skip_serialization_check=skip_serialization_check,
                        pbar=pbar.make_child(unit_progress)),
                ) for fold in folds]
        model = fit_model(model_cls, model_kwargs, all_xs, all_ys)
        pbar.progress(unit_progress)
        metrics = defaultdict(list)
        for future in metric_futures:
            _metrics = future.get()
            for k, v in _metrics.items():
                metrics[k].append(v)
        if benchmark:
            benchmark_model(model, all_xs)
    return model, metrics


def meta_repr(meta):
    meta_repr = []
    for item in meta.advice:
        meta_repr.append('{:<20} {}'.format(item.kind + ':', item.text))
    for item in meta.description:
        meta_repr.append('{:<20} {}'.format(item.heading + ':', item.text))
    return '\n'.join(meta_repr)


def fit_model(model_cls: BaseModel, model_kwargs: Dict, xs, ys) -> BaseModel:
    model = model_cls(**model_kwargs)
    model.fit(xs, ys)
    return model


def eval_on_fold(fold, model_cls: BaseModel, model_kwargs: Dict,
                 all_xs, all_ys, pbar: PBar, skip_serialization_check=False,
                 ) -> Dict:
    """ Train and evaluate the classifier on a given fold.
    """
    train_idx, test_idx = fold
    model = fit_model(model_cls, model_kwargs,
                      flt_list(all_xs, train_idx), all_ys[train_idx])
    pbar.progress(0.8)
    if not skip_serialization_check:
        model = decode_object(encode_object(model))  # type: BaseModel
    test_xs, test_ys = flt_list(all_xs, test_idx), all_ys[test_idx]
    pred_ys_prob = model.predict_proba(test_xs)[:, 1]
    pred_ys = model.predict(test_xs)
    pbar.progress(0.15)
    metrics = {
        'Accuracy': {'all': accuracy_score(test_ys, pred_ys)},
        'ROC AUC': {'all': get_roc_auc(test_ys, pred_ys_prob)},
    }

    human_idx = [idx for idx, doc in enumerate(test_xs)
                 if not doc_is_extra_sampled(doc)]
    if human_idx and len(human_idx) != len(test_idx):
        metrics['Accuracy']['human'] = \
            accuracy_score(test_ys[human_idx], pred_ys[human_idx])
        metrics['ROC AUC']['human'] = \
            get_roc_auc(test_ys[human_idx], pred_ys_prob[human_idx])
    pbar.progress(0.05)

    return metrics


def get_roc_auc(test_ys, pred_ys_prob):
    try:
        return roc_auc_score(test_ys, pred_ys_prob)
    except ValueError:
        return float('nan')


def flt_list(lst: List, indices: np.ndarray) -> List:
    # to avoid creating a big numpy array, filter the list
    indices = set(indices)
    return [x for i, x in enumerate(lst) if i in indices]


def get_domain(url: str) -> str:
    return tldextract.extract(url).registered_domain.lower()


WARN_N_RELEVANT_DOMAINS = 10
WARN_N_LABELED = 100
WARN_RELEVANT_RATIO_HIGH = 0.75
WARN_RELEVANT_RATIO_LOW = 0.05
GOOD_ROC_AUC = 0.95
WARN_ROC_AUC = 0.85
DANGER_ROC_AUC = 0.65


def get_meta(
        model: BaseModel,
        metrics: Dict[str, List[float]],
        advice: List[AdviceItem],
        all_docs: List[Dict],
        skip_eli5: bool=False,
        ) -> Meta:
    """ Return advice and a more technical model description.
    """
    advice = list(advice)
    description = []
    all_labeled = [doc for doc in all_docs if doc['relevant'] in {True, False}]
    human_labeled, extra_labeled = [], []
    for doc in all_labeled:
        (extra_labeled if doc_is_extra_sampled(doc) else human_labeled)\
            .append(doc)

    if len(human_labeled) < WARN_N_LABELED:
        advice.append(AdviceItem(
            WARNING,
            'Number of human labeled documents is just {n_human_labeled}, '
            'consider having at least {min_labeled} labeled.'
            .format(n_human_labeled=len(human_labeled),
                    min_labeled=WARN_N_LABELED)
        ))

    relevant = [doc for doc in all_labeled if doc['relevant']]
    relevant_ratio = len(relevant) / len(all_labeled)
    if relevant_ratio > WARN_RELEVANT_RATIO_HIGH:
        # This could still happen if there are too many labeled pages
        # and not enough sampled negative pages.
        advice.append(AdviceItem(
            WARNING,
            'The ratio of relevant pages is very high: {:.0%}, '
            'consider finding and labeling more non-relevant pages to improve '
            'classifier performance.'
            .format(relevant_ratio)
        ))
    if relevant_ratio < WARN_RELEVANT_RATIO_LOW:
        advice.append(AdviceItem(
            WARNING,
            'The ratio of relevant pages is very low, just {:.0%}, '
            'consider finding and labeling more relevant pages to improve '
            'classifier performance.'
            .format(relevant_ratio)
        ))

    add_quality_advice(advice, metrics)

    n_human_domains = len({get_domain(doc['url']) for doc in human_labeled})
    description.append(DescriptionItem(
        'Dataset',
        '{n_docs} documents, {n_labeled} labeled across {n_domains} domain{s}'
        '{extra_labeled}.'
        .format(
            n_docs=len(all_docs),
            n_labeled=len(human_labeled),
            n_domains=n_human_domains,
            extra_labeled='' if not extra_labeled else (
                ', {} random non-relevant documents added'
                .format(len(extra_labeled))),
            s='s' if n_human_domains > 1 else '',
        )))
    description.append(DescriptionItem(
        'Class balance',
        '{relevant:.0%} relevant, {non_relevant:.0%} not relevant{extra}.'
        .format(
            relevant=relevant_ratio,
            non_relevant=1. - relevant_ratio,
            extra='' if not extra_labeled else
            ' (including {:.0%} random non-relevant documents)'.format(
                len(extra_labeled) / len(all_labeled)),
        )))
    description.extend(metrics_description(metrics))

    if skip_eli5:
        weights = None
    else:
        weights = get_eli5_weights(model, all_labeled)

    return Meta(
        advice=advice,
        description=description,
        weights=weights,
        tooltips=TOOLTIPS,
    )


def add_quality_advice(advice, metrics):
    roc_key = 'ROC AUC'
    roc_aucs = metrics.get(roc_key)
    if not roc_aucs:
        return
    roc_auc = np.mean([m.get('human', m['all']) for m in roc_aucs])
    fix_advice = (
        'fixing warnings shown above' if advice else
        'labeling more pages, or re-labeling them using '
        'different criteria')
    if np.isnan(roc_auc):
        advice.append(AdviceItem(
            WARNING,
            'The quality of the classifier is not well defined. '
            'Consider {advice}.'
            .format(advice=fix_advice)
        ))
    elif roc_auc < WARN_ROC_AUC:
        advice.append(AdviceItem(
            WARNING,
            'The quality of the classifier is {quality}, {roc_key} is just '
            '{roc_auc:.2f}. Consider {advice}.'
            .format(
                quality=('very bad' if roc_auc < DANGER_ROC_AUC else
                         'not very good'),
                roc_key=roc_key,
                roc_auc=roc_auc,
                advice=fix_advice,
        )))
    else:
        advice.append(AdviceItem(
            NOTICE,
            'The quality of the classifier is {quality}, {roc_key} is '
            '{roc_auc:.2f}. {advice}.'
            .format(
                quality=('very good' if roc_auc > GOOD_ROC_AUC else
                         'not bad'),
                roc_key=roc_key,
                roc_auc=roc_auc,
                advice=('Still, consider fixing warnings shown above'
                        if advice else
                        'You can label more pages if you want to improve '
                        'quality, but it\'s better to start crawling and '
                        'check the quality of crawled pages'),
        )))


def metrics_description(metrics):
    description = []
    if metrics:
        description.append(DescriptionItem('Metrics', ''))
        for name, values in sorted(metrics.items()):
            all_values = [v['all'] for v in values]
            human_values = list(filter(None, (v.get('human') for v in values)))
            if human_values:
                text = (
                    '{} (human labeled examples), '
                    '{} (human labeled + random non-relevant examples)'
                    .format(format_mean_and_std(human_values),
                            format_mean_and_std(all_values)))
            else:
                text = format_mean_and_std(all_values)
            description.append(DescriptionItem(name, text))
    return description


def format_mean_and_std(values):
    return '{:.3f} ± {:.3f}'.format(np.mean(values), 1.96 * np.std(values))


def get_signed_weights(dfs):
    df = pd.concat(dfs)
    df = df.reset_index()
    df.loc[~df['target'], 'weight'] *= -1
    df = df[['feature', 'weight']]
    df_mean = df.groupby('feature').mean()
    df_mean.sort_values('weight', inplace=True)
    return df_mean


def get_eli5_weights(model: BaseModel, docs: List):
    """ Return eli5 feature weights (as a dict) with added color info.
    """
    expl = model.explain_weights()
    logging.info('explain_weights:\n{}'
                 .format(format_as_text(expl, show=fields.WEIGHTS)))
    import tqdm
    dfs = [format_as_dataframe(model.explain_prediction(doc))
           for doc in tqdm.tqdm(docs)]
    df_mean = get_signed_weights(dfs)
    logging.info('explain_prediction:\n{}'.format(repr(df_mean)))
    import IPython; IPython.embed()

    if expl.targets:
        weights = expl.targets[0].feature_weights
        weight_range = get_weight_range(weights)
        for w_lst in [weights.pos, weights.neg]:
            w_lst[:] = [{
                'feature': fw.feature,
                'weight': fw.weight,
                'hsl_color': format_hsl(weight_color_hsl(fw.weight, weight_range)),
            } for fw in w_lst]
        weights.neg.reverse()
        return format_as_dict(weights)
    elif expl.feature_importances:
        importances = expl.feature_importances.importances
        weight_range = max_or_0(abs(fw.weight) for fw in importances)
        return {
            'pos': [{
                'feature': fw.feature,
                'weight': float(fw.weight),
                'hsl_color': format_hsl(weight_color_hsl(fw.weight, weight_range)),
            } for fw in importances],
            'neg': [],
            'pos_remaining': int(expl.feature_importances.remaining),
            'neg_remaining': 0,
        }
    else:
        return {}


TOOLTIPS = {
    'ROC AUC': (
        'Area under ROC (receiver operating characteristic) curve '
        'shows how good is the classifier at telling relevant pages from '
        'non-relevant at different thresholds. '
        'Random classifier has ROC AUC = 0.5, '
        'and a perfect classifier has ROC AUC = 1.0.'
    ),
    'Accuracy': (
        'Accuracy is the ratio of pages classified correctly as '
        'relevant or not relevant. This metric is easy to interpret but '
        'not very good for unbalanced datasets.'
    ),
}


def benchmark_model(model, xs):
    n_batch = 100
    n_single = 20
    xs = np.array(_exactly_n_items(xs, n_batch))
    batch_time = mean(_time(model.predict_proba, xs) / n_batch
                      for _ in range(4))
    single_time = mean(_time(model.predict_proba, np.array([x]))
                       for x in _exactly_n_items(xs, n_single))
    logging.info(
        'Prediction performance (without text extraction): '
        '{batched:.1f} rps 100-batched, {single:.1f} rps single'
        .format(batched=1 / batch_time, single=1 / single_time))


def _exactly_n_items(lst, n):
    assert len(lst) > 0
    result = []
    while True:
        for x in lst:
            result.append(x)
            if len(result) == n:
                return result


def _time(fn, *args, **kwargs):
    t0 = time.time()
    fn(*args, **kwargs)
    return time.time() - t0


def train_model_cli(message_filename, args):
    opener = gzip.open if message_filename.endswith('.gz') else open
    with opener(message_filename, 'rt') as f:
        logging.info('Decoding message')
        message = json.load(f)
    logging.info('Done, starting train_model')
    t0 = time.time()
    result = train_model(
        message['pages'],
        skip_eli5=args.no_eli5,
        skip_serialization_check=args.no_dump,
        lda=args.lda,
        doc2vec=args.doc2vec,
        dmoz_fasttext=args.dmoz_fasttext,
        dmoz_sklearn=args.dmoz_sklearn,
        clf_kind=args.clf,
        random_pages=args.random_pages,
        benchmark=True,
        skip_validation=args.no_validation,
        model_cls=getattr(models, args.model) if args.model else None,
    )
    logging.info('Training took {:.1f} s'.format(time.time() - t0))
    logging.info(
        'Model size: {:,} bytes'.format(len(encode_object(result.model))))
    return result


def main():
    configure_logging()

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('message_filenames', nargs='+')
    arg('--clf', choices=sorted(models.DefaultModel.clf_kinds))
    arg('--model', choices=['DefaultModel', 'CharModel'])
    arg('--no-dump', action='store_true', help='skip serialization checks')
    arg('--no-eli5', action='store_true', help='skip eli5')
    arg('--no-validation', action='store_true', help='skip validation')
    arg('--random-pages', help='path to random negative pages',
        default='random-pages.jl.gz')
    arg('--lda', help='path to LDA model')
    arg('--doc2vec', help='path to doc2vec model')
    arg('--dmoz-fasttext', help='path to dmoz fasttext model')
    arg('--dmoz-sklearn', help='path to dmoz sklearn model in .pkl format')
    args = parser.parse_args()

    train_results = [
        train_model_cli(filename, args) for filename in args.message_filenames]
    if len(train_results) > 1:
        for filename, result in zip(args.message_filenames, train_results):
            print('\nResults for {}'.format(filename))
            print(meta_repr(result.meta))

        print()
        metric = 'ROC AUC'
        for kind in ['all', 'human']:
            logging.info('gmean {metric} of "{kind}" runs: {gmean:.3f}'.format(
                metric=metric,
                kind=kind,
                gmean=gmean([np.mean([run[kind] for run in res.metrics[metric]])
                             for res in train_results]),
            ))

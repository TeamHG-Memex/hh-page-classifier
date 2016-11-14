from collections import defaultdict, namedtuple
from functools import partial
import logging
import multiprocessing
from pprint import pformat
from typing import Any, List, Dict, Tuple

import attr
from eli5.sklearn.explain_weights import explain_weights
import html_text
import numpy as np
from sklearn.cross_validation import LabelKFold, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import tldextract


ERROR = 'Error'
WARNING = 'Warning'
NOTICE = 'Notice'

MetaItem = Tuple[str, str]

@attr.s
class Meta:
    advice = attr.ib()  # type: List[MetaItem]
    description = attr.ib(default=None)  # type: List[MetaItem]

    def asdict(self):
        return attr.asdict(self, retain_collection_types=False)


ModelMeta = namedtuple('ModelMeta', 'model, meta')


def train_model(docs: List[Dict], fit_clf=None) -> ModelMeta:
    """ Train and evaluate a model.
    docs is a list of dicts:
    {'url': url, 'html': html, 'relevant': True/False/None}.
    Return the model itself and a human-readable description of it's performance.
    """
    fit_clf = fit_clf or default_fit_clf
    if not docs:
        return ModelMeta(
            model=None,
            meta=Meta(
                [(ERROR, 'Can not train a model: no pages given.')]))
    with_labels = [doc for doc in docs if doc.get('relevant') in [True, False]]
    if not with_labels:
        return ModelMeta(
            model=None,
            meta=Meta(
                [(ERROR, 'Can not train a model, no labeled pages given.')]))

    all_ys = np.array([doc['relevant'] for doc in with_labels])
    classes = np.unique(all_ys)
    if len(classes) == 1:
        only_cls = classes[0]
        class_names = ['not relevant', 'relevant']
        return ModelMeta(
            model=None,
            meta=Meta(
                [(ERROR,
                  'Can not train a model, only {} pages in sample: '
                  'need examples of {} pages too.'.format(
                      class_names[only_cls], class_names[not only_cls])),
                 ]))

    logging.info('Extracting text')
    with multiprocessing.Pool() as pool:
        all_xs = pool.map(html_text.extract_text,
                          [doc['html'] for doc in with_labels],
                          chunksize=100)

    logging.info('Training and evaluating model')
    metrics = defaultdict(list)
    domains = [get_domain(doc['url']) for doc in with_labels]
    n_domains = len(set(domains))
    n_folds = 4
    advice = []
    if n_domains == 1:
        advice.append((
            WARNING,
            'Only 1 domain in data means that it\'s impossible to do '
            'cross-validation across domains, '
            'and might result in model over-fitting.'
        ))
        folds = KFold(len(all_xs), n_folds=n_folds)
    else:
        folds = LabelKFold(domains, n_folds=min(n_domains, n_folds))
        if n_domains < n_folds:
            advice.append((
                WARNING,
                'Low number of domains (just {}) '
                'might result in model over-fitting.'.format(n_domains)
            ))
    folds = [fold for fold in folds if len(np.unique(all_ys[fold[0]])) > 1]
    with multiprocessing.Pool() as pool:
        clf_future = pool.apply_async(fit_clf, args=(all_xs, all_ys))
        if folds:
            for _metrics in pool.imap_unordered(
                    partial(eval_on_fold,
                            all_xs=all_xs, all_ys=all_ys, fit_clf=fit_clf),
                    folds):
                for k, v in _metrics.items():
                    metrics[k].append(v)
        else:
            advice.append((
                WARNING,
                'Can not do cross-validation, as there are no folds where '
                'training data has both relevant and non-relevant examples. '
                'There are too few domains or the dataset is too unbalanced.'
            ))
        clf = clf_future.get()

    meta = get_meta(clf, metrics, advice, docs, with_labels, n_domains)
    logging.info('Model meta:\n{}'.format(pformat(attr.asdict(meta))))
    return ModelMeta(model=clf, meta=meta)


def default_fit_clf(xs, ys) -> Pipeline:
    # Doing explicit feature selection because:
    # - eli5 does not support pipelines with feature selection
    # - final model should be small and does not need the whole vocabulary
    vec = TfidfVectorizer()
    transformed = vec.fit_transform(xs)
    feature_selection_clf = SGDClassifier(
        loss='log', penalty='l2', n_iter=50, random_state=42)
    feature_selection_clf.fit(transformed, ys)
    abs_coefs = np.abs(feature_selection_clf.coef_[0])
    features = set((abs_coefs > np.mean(abs_coefs)).nonzero()[0])
    # FIXME - relies on ngram_range=(1, 1) ?
    vocabulary = [w for w, idx in vec.vocabulary_.items() if idx in features]
    clf = Pipeline([
        ('vec', TfidfVectorizer(vocabulary=vocabulary)),
        ('clf', LogisticRegressionCV(random_state=42)),
    ])
    clf.fit(xs, ys)
    return clf


def eval_on_fold(fold, all_xs, all_ys, fit_clf) -> Dict:
    """ Train and evaluate the classifier on a given fold.
    """
    train_idx, test_idx = fold
    clf = fit_clf(flt_list(all_xs, train_idx), all_ys[train_idx])
    test_xs, test_ys = flt_list(all_xs, test_idx), all_ys[test_idx]
    pred_ys_prob = clf.predict_proba(test_xs)[:, 1]
    pred_ys = clf.predict(test_xs)
    try:
        auc = roc_auc_score(test_ys, pred_ys_prob)
    except ValueError:
        auc = float('nan')
    return {
        'Accuracy': accuracy_score(test_ys, pred_ys),
        'F1': f1_score(test_ys, pred_ys),
        'ROC AUC': auc,
    }


def flt_list(lst: List, indices: np.ndarray) -> List:
    # to avoid creating a big numpy array, filter the list
    indices = set(indices)
    return [x for i, x in enumerate(lst) if i in indices]


WARN_N_LABELED = 100
WARN_RELEVANT_RATIO_HIGH = 0.75
WARN_RELEVANT_RATIO_LOW = 0.05
GOOD_ROC_AUC = 0.95
WARN_ROC_AUC = 0.85
DANGER_ROC_AUC = 0.65


def get_meta(
        clf: Pipeline,
        metrics: Dict[str, List[float]],
        advice: List[MetaItem],
        docs: List[Dict],
        with_labels: List[Dict],
        n_domains: int,
        ) -> Meta:
    """ Return advice and a more technical model description.
    """
    advice = list(advice)
    description = []
    n_docs = len(docs)
    n_labeled = len(with_labels)
    relevant = [doc for doc in docs if doc['relevant']]
    relevant_ratio = len(relevant) / n_labeled

    if n_labeled < WARN_N_LABELED:
        advice.append((
            WARNING,
            'Number of labeled documents is just {n_labeled}, '
            'consider having at least {min_labeled} labeled.'
            .format(n_labeled=n_labeled, min_labeled=WARN_N_LABELED)
        ))
    if relevant_ratio > WARN_RELEVANT_RATIO_HIGH:
        advice.append((
            WARNING,
            'The ratio of relevant pages is very high: {:.0%}, '
            'consider finding and labeling more irrelevant pages to improve '
            'classifier performance.'
        ))
    if relevant_ratio < WARN_RELEVANT_RATIO_LOW:
        advice.append((
            WARNING,
            'The ratio of relevant pages is very low, just {:.0%}, '
            'consider finding and labeling more relevant pages to improve '
            'classifier performance.'
        ))
    roc_aucs = metrics.get('ROC AUC')
    if roc_aucs:
        roc_auc = np.mean(roc_aucs)
        if roc_auc < WARN_ROC_AUC:
            advice.append((
                WARNING,
                'The quality of classifier is {quality}, ROC AUC is just '
                '{roc_auc:.2f}. Consider {advice}.'
                .format(
                    quality=('very bad' if roc_auc < DANGER_ROC_AUC else
                             'not very good'),
                    roc_auc=roc_auc,
                    advice=('fixing warnings shown above' if advice else
                            'labeling more pages, or re-labeling them using '
                            'different criteria'),
                )
            ))
        else:
            advice.append((
                NOTICE,
                'The quality of classifier is {quality}, ROC AUC is '
                '{roc_auc:.2f}. {advice}.'
                .format(
                    quality=('very good' if roc_auc > GOOD_ROC_AUC else
                             'not bad'),
                    roc_auc=roc_auc,
                    advice=('Still, consider fixing warnings shown above'
                            if advice else
                            'You can label more pages if you want to improve '
                            'quality, but it\'s better to start crawling and '
                            'check the quality of crawled pages.'),
                )
            ))

    description.append((
        'Dataset',
        '{n_docs} documents, {n_labeled} with labels ({labeled_ratio:.0%}) '
        'across {n_domains} domain{s}.'.format(
            n_docs=n_docs,
            n_labeled=n_labeled,
            labeled_ratio=n_labeled / n_docs,
            n_domains=n_domains,
            s='s' if n_domains > 1 else '',
        )))
    description.append((
        'Class balance',
        '{:.0%} relevant, {:.0%} not relevant.'
        .format(relevant_ratio, 1. - relevant_ratio)))
    if metrics:
        description.append(('Metrics', ''))
        description.extend(
            (k, '{:.3f} ± {:.3f}'.format(np.mean(v), 1.96 * np.std(v)))
            for k, v in sorted(metrics.items()))

    weights_explanation = explain_weights(
        clf.named_steps['clf'], vec=clf.named_steps['vec'], top=30)
    fw = weights_explanation.targets[0].feature_weights
    description.extend(features_descr(fw.pos, fw.pos_remaining, 'Positive'))
    description.extend(features_descr(fw.neg, fw.neg_remaining, 'Negative'))

    return Meta(advice=advice, description=description)


def features_descr(features: List[Tuple[str, float]],
                   remaining_features: int, group_name: str) -> List[MetaItem]:
    description = []
    if features or remaining_features:
        description.append(('{} features'.format(group_name), ''))
        description.extend((str(feature), '{:.2f}'.format(weight))
                           for feature, weight in features)
        if remaining_features:
            description.append(('Other {} features'.format(group_name.lower()),
                                str(remaining_features)))
    return description


def get_domain(url: str) -> str:
    return tldextract.extract(url).registered_domain.lower()


def main():
    import argparse
    import gzip
    import json
    import time
    from .utils import configure_logging
    from .service import encode_model

    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('message_filename')
    args = parser.parse_args()
    opener = gzip.open if args.message_filename.endswith('.gz') else open
    with opener(args.message_filename, 'rt') as f:
        logging.info('Decoding message')
        message = json.load(f)
    logging.info('Done, starting train_model')
    t0 = time.time()
    result = train_model(message['pages'])
    logging.info('Training took {:.1f} s'.format(time.time() - t0))
    logging.info('Model size: {:,} bytes'.format(len(encode_model(result.model))))

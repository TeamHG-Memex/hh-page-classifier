from collections import defaultdict, namedtuple
import logging
from typing import List, Dict

from eli5.sklearn.explain_weights import explain_weights
import html_text
import numpy as np
from sklearn.cross_validation import LabelKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import tldextract


ModelMeta = namedtuple('ModelMeta', 'model, meta')


def train_model(docs: List[Dict]) -> ModelMeta:
    """ Train and evaluate a model.
    docs is a list of dicts:
    {'url': url, 'html': html, 'relevant': True/False/None}.
    Return the model itself and a human-readable description of it's performance.
    """
    with_labels = [doc for doc in docs if doc.get('relevant') in [True, False]]
    all_xs = [html_text.extract_text(doc['html']) for doc in with_labels]
    all_ys = np.array([doc['relevant'] for doc in with_labels])
    classes = np.unique(all_ys)
    if len(classes) < 2:
        only_cls = classes[0]
        class_names = ['not relevant', 'relevant']
        return ModelMeta(
            model=None,
            meta='Only {} pages in sample: need examples of {} pages too.'.format(
                class_names[only_cls], class_names[not only_cls]))
    logging.info('Evaluating model')
    metrics = defaultdict(list)
    for train_idx, test_idx in LabelKFold(
            [get_domain(doc['url']) for doc in with_labels], n_folds=4):
        if len(np.unique(all_ys[train_idx])) == 2:
            for k, v in eval_on_fold(all_xs, all_ys, train_idx, test_idx).items():
                metrics[k].append(v)
    logging.info('Training final model')
    clf = init_clf()
    clf.fit(all_xs, all_ys)
    return ModelMeta(model=clf, meta=describe_model(clf, metrics, docs))


def init_clf() -> Pipeline:
    return Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegressionCV(random_state=42)),
    ])


def eval_on_fold(all_xs, all_ys, train_idx, test_idx) -> Dict:
    """ Train and evaluate the classifier on a given fold.
    """
    clf = init_clf()
    clf.fit(flt_list(all_xs, train_idx), all_ys[train_idx])
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


def describe_model(clf: Pipeline, metrics: Dict, docs: List[Dict]) -> str:
    """ Return a human-readable model description.
    """
    descr = []
    descr += ['Metrics:']
    aggr_metrics = {
        k: '  {:.3f} ± {:.3f}'.format(np.mean(v), 1.96 * np.std(v))
        for k, v in metrics.items()}
    descr += ['{:<20}: {}'.format(k, v) for k, v in sorted(aggr_metrics.items())]
    with_labels = [doc for doc in docs if doc.get('relevant') in [True, False]]
    descr += ['Dataset: {n_docs} documents, {labeled_ratio:.0%} with labels'
              .format(n_docs=len(docs),
                      labeled_ratio=len(with_labels) / len(docs))]
    relevant = [doc for doc in docs if doc['relevant']]
    relevant_ratio = len(relevant) / len(with_labels)
    descr += ['Class balance: {:.0%} relevant, {:.0%} not relevant'
              .format(relevant_ratio, 1. - relevant_ratio)]
    weights_explanation = explain_weights(
        clf.named_steps['clf'], vec=clf.named_steps['vect'], top=10)
    feature_weights = weights_explanation['classes'][0]['feature_weights']
    descr.extend(features_descr(feature_weights, 'pos', 'Positive'))
    descr.extend(features_descr(feature_weights, 'neg', 'Negative'))
    # TODO - some advice: are metrics good or bad, how is the class balance
    return '\n'.join(descr)


def features_descr(feature_weights, key, key_name):
    rem_key = '{}_remaining'.format(key)
    descr = []
    if key in feature_weights or rem_key in feature_weights:
        descr += ['{} features:'.format(key_name)]
        descr.extend('{:<20}: {:.2f}'.format(feature, weight)
                     for feature, weight in feature_weights.get(key, []))
        remaining = feature_weights.get(rem_key)
        if remaining:
            descr += ['Other {} features: {}'.format(key_name.lower(), remaining)]
    return descr


def get_domain(url: str) -> str:
    return tldextract.extract(url).registered_domain.lower()



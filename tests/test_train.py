from functools import partial

from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV

from hh_page_clf.train import train_model as default_train_model


def fit_clf(xs, ys) -> Pipeline:
    clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegressionCV(random_state=42)),
    ])
    clf.fit(xs, ys)
    return clf


train_model = partial(default_train_model, fit_clf=fit_clf)


def test_train_model():
    data = fetch_20newsgroups(
        random_state=42,
        categories=['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'])
    limit = None
    if limit is not None:
        data['target'] = data['target'][:limit]
        data['data'] = data['data'][:limit]
    n_domains = int(len(data['target']) / 5)
    docs = [
        {
            'html': '\n'.join('<p>{}</p>'.format(t) for t in text.split('\n')),
            'url': 'http://example-{}.com/{}'.format(n % n_domains, n),
            'relevant': {'sci.space': True, 'sci.med': None}.get(
                data['target_names'][target], False),
        }
        for n, (text, target) in enumerate(zip(data['data'], data['target']))]
    result = train_model(docs)
    print(result.meta)
    assert result.meta == '''\
Dataset: 2373 documents, 75% with labels across 473 domains.
Class balance: 33% relevant, 67% not relevant.
Metrics:
Accuracy            :   0.983 ± 0.008
F1                  :   0.973 ± 0.011
ROC AUC             :   0.999 ± 0.002
Positive features:
space               : 37.16
orbit               : 14.87
moon                : 13.75
launch              : 12.92
dc                  : 11.78
nasa                : 11.31
earth               : 10.79
rockets             : 10.48
Other positive features: 11506
Negative features:
chip                : -12.32
key                 : -10.66
Other negative features: 18105'''
    assert isinstance(result.model, Pipeline)
    assert hasattr(result.model, 'predict_proba')


def test_empty():
    result = train_model([])
    assert result.meta == 'Can not train a model: no pages given.'
    assert result.model is None


def test_unlabeled():
    docs = [{'html': 'foo',
             'url': 'http://example{}.com'.format(i),
             'relevant': None}
            for i in range(10)]
    result = train_model(docs)
    assert result.meta == 'Can not train a model: no labeled pages given.'
    assert result.model is None


def test_unbalanced():
    docs = [{'html': 'foo',
             'url': 'http://example{}.com'.format(i),
             'relevant': True}
            for i in range(10)]
    result = train_model(docs)
    assert result.meta == (
        'Can not train a model. Only relevant pages in sample: '
        'need examples of not relevant pages too.')
    assert result.model is None

    docs = [{'html': 'foo',
             'url': 'http://example{}.com'.format(i),
             'relevant': False}
            for i in range(10)]
    result = train_model(docs)
    assert result.meta == (
        'Can not train a model. Only not relevant pages in sample: '
        'need examples of relevant pages too.')
    assert result.model is None


def test_single_domain():
    docs = [{'html': 'foo{} bar'.format(i % 4),
             'url': 'http://example.com/{}'.format(i),
             'relevant': i % 2 == 0}
            for i in range(10)]
    result = train_model(docs)
    assert result.meta == """\
Warning: only 1 domain in data means that it's impossible to do cross-validation across domains, and might result in model over-fitting.
Dataset: 10 documents, 100% with labels across 1 domain.
Class balance: 50% relevant, 50% not relevant.
Metrics:
Accuracy            :   1.000 ± 0.000
F1                  :   1.000 ± 0.000
ROC AUC             :   1.000 ± 0.000
Positive features:
foo0                : 0.00
foo2                : 0.00
Negative features:
foo1                : -0.00
foo3                : -0.00"""
    assert result.model is not None


def test_two_domains_bad_folds():
    docs = [{'html': 'foo{}'.format(i % 4),
             'url': 'http://example{}.com/{}'.format(i % 2, i),
             'relevant': i % 2 == 0}
            for i in range(10)]
    result = train_model(docs)
    assert result.meta == """Warning: low number of domains (just 2) might result in model over-fitting.
Warning: Can not do cross-validation, as there are no folds where training data has both relevant and non-relevant examples. There are too few domains or the dataset is too unbalanced.
Dataset: 10 documents, 100% with labels across 2 domains.
Class balance: 50% relevant, 50% not relevant.
Positive features:
foo0                : 0.00
foo2                : 0.00
Negative features:
foo1                : -0.00
foo3                : -0.00"""
    assert result.model is not None


def test_two_domains():
    docs = [{'html': 'foo{}'.format(i % 3),
             'url': 'http://example{}.com/{}'.format(i % 2, i),
             'relevant': i % 3 == 0}
            for i in range(10)]
    result = train_model(docs)
    print(result.meta)
    assert result.meta == """Warning: low number of domains (just 2) might result in model over-fitting.
Dataset: 10 documents, 100% with labels across 2 domains.
Class balance: 40% relevant, 60% not relevant.
Metrics:
Accuracy            :   1.000 ± 0.000
F1                  :   1.000 ± 0.000
ROC AUC             :   1.000 ± 0.000
Positive features:
foo0                : 2.19
Negative features:
foo2                : -1.10
foo1                : -1.10
<BIAS>              : -0.79"""
    assert result.model is not None


def test_default_clf():
    docs = [{'html': 'foo{} bar'.format(i % 4),
             'url': 'http://example{}.com'.format(i),
             'relevant': i % 2 == 0}
            for i in range(10)]
    result = train_model(docs)
    assert result.model is not None
    assert result.meta.startswith("""\
Dataset: 10 documents, 100% with labels across 10 domains.
Class balance: 50% relevant, 50% not relevant.
Metrics:""")

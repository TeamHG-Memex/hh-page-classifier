from functools import partial
import json
from pprint import pprint

import attr
from eli5.sklearn.explain_weights import explain_weights
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from hh_page_clf.model import BaseModel, get_attributes, set_attributes
from hh_page_clf.train import train_model as default_train_model, Meta, AdviceItem


class ATestModel(BaseModel):
    def __init__(self):
        self.vec = CountVectorizer(preprocessor=lambda x: x['text'].lower())
        self.clf = LogisticRegression(random_state=42)
        self.pipeline = make_pipeline(self.vec, self.clf)
        super().__init__()

    def get_params(self):
        return {'vec_attrs': get_attributes(self.vec),
                'clf_attrs': get_attributes(self.clf),
                }

    def set_params(self, *, vec_attrs, clf_attrs):
        set_attributes(self, 'vec', vec_attrs)
        set_attributes(self, 'clf', clf_attrs)

    def fit(self, xs, ys):
        self.pipeline.fit(xs, ys)

    def predict(self, xs):
        return self.pipeline.predict(xs)

    def predict_proba(self, xs):
        return self.pipeline.predict_proba(xs)

    def explain_weights(self):
        return explain_weights(self.clf, vec=self.vec, top=30)


train_model = partial(default_train_model, model_cls=ATestModel)


def lst_as_dict(lst):
    return [attr.asdict(x) for x in lst]


def test_train_model():
    data = fetch_20newsgroups(
        random_state=42,
        categories=['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'])
    limit = 200
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
    pprint(attr.asdict(result.meta))
    assert lst_as_dict(result.meta.advice) == [
        {'kind': 'Notice',
         'text': "The quality of the classifier is very good, ROC AUC is 0.96. "
                 "You can label more pages if you want to improve quality, "
                 "but it's better to start crawling "
                 "and check the quality of crawled pages.",
         },
        ]
    assert lst_as_dict(result.meta.description) == [
        {'heading': 'Dataset',
         'text': '200 documents, 159 labeled across 40 domains.'},
        {'heading': 'Class balance',
         'text': '33% relevant, 67% not relevant.'},
        {'heading': 'Metrics', 'text': ''},
        {'heading': 'Accuracy', 'text': '0.881 ± 0.122'},
        {'heading': 'ROC AUC', 'text': '0.964 ± 0.081'}]
    assert len(result.meta.weights['pos']) > 0
    assert len(result.meta.weights['neg']) > 0
    assert isinstance(result.model, BaseModel)
    assert hasattr(result.model, 'predict_proba')


def test_empty():
    result = train_model([])
    pprint(attr.asdict(result.meta))
    assert result.meta == Meta(
        advice=[AdviceItem('Error', 'Can not train a model, no pages given.')])
    assert result.model is None


def test_unlabeled():
    docs = [{'html': 'foo',
             'url': 'http://example{}.com'.format(i),
             'relevant': None}
            for i in range(10)]
    result = train_model(docs)
    pprint(attr.asdict(result.meta))
    assert result.meta == Meta(
        advice=[AdviceItem(
            'Error', 'Can not train a model, no labeled pages given.')])
    assert result.model is None


def test_unbalanced():
    docs = [{'html': 'foo',
             'url': 'http://example{}.com'.format(i),
             'relevant': True}
            for i in range(10)]
    result = train_model(docs)
    pprint(attr.asdict(result.meta))
    assert result.meta == Meta(
        advice=[AdviceItem(
            'Error',
            'Can not train a model, only relevant pages in sample: '
            'need examples of not relevant pages too.')])
    assert result.model is None

    docs = [{'html': 'foo',
             'url': 'http://example{}.com'.format(i),
             'relevant': False}
            for i in range(10)]
    result = train_model(docs)
    pprint(attr.asdict(result.meta))
    assert result.meta == Meta(
        advice=[AdviceItem(
            'Error',
            'Can not train a model, only not relevant pages in sample: '
            'need examples of relevant pages too.')])
    assert result.model is None


def test_single_domain():
    docs = [{'html': 'foo{} bar'.format(i % 4),
             'url': 'http://example.com/{}'.format(i),
             'relevant': i % 2 == 0}
            for i in range(10)]
    result = train_model(docs)
    pprint(attr.asdict(result.meta))
    assert lst_as_dict(result.meta.advice)[:2] == [
        {'kind': 'Warning',
         'text': "Only 1 relevant domain in data means that it's impossible to do "
                 'cross-validation across domains, and will likely result in '
                 'model over-fitting.'},
        {'kind': 'Warning',
         'text': 'Number of human labeled documents is just 10, consider having '
                 'at least 100 labeled.'},
    ]
    assert lst_as_dict(result.meta.description)[:3] == [
        {'heading': 'Dataset',
         'text': '10 documents, 10 labeled across 1 domain.'},
        {'heading': 'Class balance',
         'text': '50% relevant, 50% not relevant.'},
        {'heading': 'Metrics', 'text': ''},
    ]
    assert result.model is not None


def test_two_domains():
    docs = [{'html': 'foo{}'.format(i % 3),
             'url': 'http://example{}.com/{}'.format(i % 2, i),
             'relevant': i % 3 == 0}
            for i in range(10)]
    result = train_model(docs)
    pprint(attr.asdict(result.meta))
    assert lst_as_dict(result.meta.advice) == [
        {'kind': 'Warning',
         'text': 'Low number of relevant domains (just 2) might result in model '
                 'over-fitting.'},
        {'kind': 'Warning',
         'text': 'Number of human labeled documents is just 10, consider having '
                 'at least 100 labeled.'},
        {'kind': 'Notice',
         'text': 'The quality of the classifier is very good, ROC AUC is '
                 '1.00. Still, consider fixing warnings shown above.'}]
    assert lst_as_dict(result.meta.description) == [
        {'heading': 'Dataset',
         'text': '10 documents, 10 labeled across 2 domains.'},
        {'heading': 'Class balance',
         'text': '40% relevant, 60% not relevant.'},
        {'heading': 'Metrics', 'text': ''},
        {'heading': 'Accuracy', 'text': '1.000 ± 0.000'},
        {'heading': 'ROC AUC', 'text': '1.000 ± 0.000'}]
    assert result.model is not None


def test_default_clf():
    docs = [{'html': 'foo{} bar'.format(i % 4),
             'url': 'http://example{}.com'.format(i),
             'relevant': i % 2 == 0}
            for i in range(10)]
    result = default_train_model(docs)
    assert result.model is not None
    meta = attr.asdict(result.meta)
    pprint(meta)
    json.dumps(meta)

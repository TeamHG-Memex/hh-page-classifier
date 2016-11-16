from typing import Dict, Any

from eli5.sklearn.explain_weights import explain_weights
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.pipeline import make_pipeline, make_union

from .utils import encode_object, decode_object


class BaseModel:
    @classmethod
    def decode(cls, data: str) -> 'BaseModel':
        params = decode_object(data)
        assert isinstance(params, dict)
        return cls(**params)

    def encode(self) -> str:
        return encode_object(self.get_params())

    def get_params(self) -> Dict[str, Any]:
        raise NotImplementedError

    def fit(self, xs, yx) -> None:
        raise NotImplementedError

    def predict(self, xs):
        raise NotImplementedError

    def predict_proba(self, xs):
        raise NotImplementedError

    def explain_weights(self):
        raise NotImplementedError


class DefaultModel(BaseModel):
    def __init__(self, text_vec_attrs=None, url_vec_attrs=None, clf_attrs=None):
        self.default_text_preprocessor = TfidfVectorizer().build_preprocessor()
        self.text_vec = TfidfVectorizer(
            preprocessor=self.text_preprocessor,
        )
        if text_vec_attrs:
            self.text_vec._tfidf._idf_diag = text_vec_attrs['_idf_diag']
            self.text_vec.vocabulary_ = text_vec_attrs['vocabulary_']
        self.url_vec = CountVectorizer(
            binary=True,
            analyzer='char',
            ngram_range=(3, 4),
            preprocessor=self.url_preprocessor,
        )
        if url_vec_attrs:
            set_attributes(self.url_vec, url_vec_attrs)
        self.vec = make_union(self.text_vec, self.url_vec)
        self.clf = LogisticRegressionCV(random_state=42)
        if clf_attrs:
            set_attributes(self.clf, clf_attrs)
        self.pipeline = make_pipeline(self.vec, self.clf)

    def text_preprocessor(self, item):
        return self.default_text_preprocessor(item['text'])

    def url_preprocessor(self, item):
        return item['url'].lower()

    def fit(self, xs, ys):
        vec = TfidfVectorizer(preprocessor=self.text_preprocessor)
        transformed = vec.fit_transform(xs)
        feature_selection_clf = SGDClassifier(
            loss='log', penalty='l2', n_iter=50, random_state=42)
        feature_selection_clf.fit(transformed, ys)
        abs_coefs = np.abs(feature_selection_clf.coef_[0])
        features = set((abs_coefs > np.mean(abs_coefs)).nonzero()[0])
        # FIXME - relies on ngram_range=(1, 1)
        self.text_vec.vocabulary = [
            w for w, idx in vec.vocabulary_.items() if idx in features]
        self.pipeline.fit(xs, ys)

    def predict(self, xs):
        return self.pipeline.predict(xs)

    def predict_proba(self, xs):
        return self.pipeline.predict_proba(xs)

    def explain_weights(self):
        return explain_weights(self.clf, vec=self.vec, top=30)

    def get_params(self):
        return {
            'text_vec_attrs': {
                '_idf_diag': self.text_vec._tfidf._idf_diag,
                'vocabulary_': self.text_vec.vocabulary_,
            },
            'url_vec_attrs': get_attributes(self.url_vec),
            'clf_attrs': get_attributes(self.clf),
        }


def get_attributes(obj):
    return {attr: getattr(obj, attr) for attr in dir(obj)
            if not attr.startswith('_') and attr.endswith('_')}


def set_attributes(obj, attributes):
    for k, v in attributes.items():
        setattr(obj, k, v)

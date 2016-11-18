from typing import Dict, Any

from eli5.sklearn.explain_weights import explain_weights
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline, FeatureUnion


class BaseModel:
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def get_params(self) -> Dict[str, Any]:
        raise NotImplementedError

    def set_params(self, **params) -> None:
        raise NotImplementedError

    def fit(self, xs, ys) -> None:
        raise NotImplementedError

    def predict(self, xs):
        raise NotImplementedError

    def predict_proba(self, xs):
        raise NotImplementedError

    def explain_weights(self):
        raise NotImplementedError

    def __getstate__(self):
        params = self.get_params()
        params['_kwargs'] = self._kwargs
        return params

    def __setstate__(self, state):
        kwargs = state.pop('_kwargs', {})
        self.__init__(**kwargs)
        self.set_params(**state)


class DefaultModel(BaseModel):
    clf_kinds = {
        'logcv': lambda: LogisticRegressionCV(random_state=42),
        'extra_tree': lambda : ExtraTreesClassifier(
            n_estimators=100, random_state=42),
    }

    def __init__(self, use_url=True, clf_kind='logcv'):
        self.default_text_preprocessor = TfidfVectorizer().build_preprocessor()
        self.text_vec = TfidfVectorizer(preprocessor=self.text_preprocessor)
        vectorizers = [('text', self.text_vec)]
        if use_url:
            self.url_vec = CountVectorizer(
                binary=True,
                analyzer='char',
                ngram_range=(3, 4),
                preprocessor=self.url_preprocessor,
            )
            vectorizers.append(('url', self.url_vec))
        else:
            self.url_vec = None
        self.vec = FeatureUnion(vectorizers)
        self.clf = self.clf_kinds[clf_kind]()
        self.pipeline = make_pipeline(self.vec, self.clf)
        super().__init__(use_url=use_url)

    def text_preprocessor(self, item):
        return self.default_text_preprocessor(item['text'])

    @staticmethod
    def url_preprocessor(item):
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
        # TODO - ideally, get features for both vectorizers
        expl = explain_weights(
            self.clf, vec=self.vec, top=30, feature_re='(^text__|^<BIAS>$)')
        fweights = expl.targets[0].feature_weights
        for fw_lst in [fweights.pos, fweights.neg]:
            for fw in fw_lst:
                if fw.feature.startswith('text__'):
                    fw.feature = fw.feature[len('text__'):]
        return expl

    def get_params(self):
        return {
            'text_vec_attrs': {
                '_idf_diag': self.text_vec._tfidf._idf_diag,
                'vocabulary_': self.text_vec.vocabulary_,
            },
            'url_vec_attrs': get_attributes(self.url_vec),
            'clf_attrs': get_attributes(self.clf),
        }

    def set_params(self, *, text_vec_attrs, url_vec_attrs, clf_attrs):
        self.text_vec._tfidf._idf_diag = text_vec_attrs['_idf_diag']
        self.text_vec.vocabulary_ = text_vec_attrs['vocabulary_']
        set_attributes(self.url_vec, url_vec_attrs)
        set_attributes(self.clf, clf_attrs)


def get_attributes(obj):
    return {attr: getattr(obj, attr) for attr in dir(obj)
            if not attr.startswith('_') and attr.endswith('_')}


def set_attributes(obj, attributes):
    for k, v in attributes.items():
        setattr(obj, k, v)

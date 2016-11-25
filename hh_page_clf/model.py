from typing import Dict, Any

from eli5.sklearn.explain_weights import explain_weights
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer


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
    default_clf_kind = 'logcv'

    def __init__(self, use_url=True, use_lda=True, clf_kind=default_clf_kind):
        vectorizers = []
        if use_url:
            self.url_vec = TfidfVectorizer(
                analyzer='char',
                ngram_range=(3, 4),
                preprocessor=self.url_preprocessor,
            )
            vectorizers.append(('url', self.url_vec))
        else:
            self.url_vec = None
        if use_lda:
            self.lda = joblib.load('lda-15k.joblib')
            # TODO - proper feature names
            vectorizers.append(
                ('lda', FunctionTransformer(
                    func=lambda xs: self.lda.transform(
                        [x['text'].lower() for x in xs]),
                    validate=False,
                )))
        self.default_text_preprocessor = TfidfVectorizer().build_preprocessor()
        self.text_vec = TfidfVectorizer(preprocessor=self.text_preprocessor)
        vectorizers.append(('text', self.text_vec))
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
        expl = explain_weights(self.clf, vec=self.vec, top=50)
        fweights = expl.targets[0].feature_weights
        for fw_lst in [fweights.pos, fweights.neg]:
            for fw in fw_lst:
                if fw.feature.startswith('text__'):
                    fw.feature = fw.feature[len('text__'):]
                elif fw.feature.startswith('url__'):
                    fw.feature = 'url: {}'.format(fw.feature[len('url__'):])
        return expl

    def get_params(self):
        return {
            'text_vec_attrs': get_attributes(self.text_vec),
            'url_vec_attrs': get_attributes(self.url_vec),
            'clf_attrs': get_attributes(self.clf),
        }

    def set_params(self, *, text_vec_attrs, url_vec_attrs, clf_attrs):
        set_attributes(self.text_vec, text_vec_attrs)
        set_attributes(self.url_vec, url_vec_attrs)
        set_attributes(self.clf, clf_attrs)


class LDAModel(BaseModel):
    def __init__(self):
        self.lda = joblib.load('lda-15k.joblib')
        self.clf = ExtraTreesClassifier(n_estimators=100)
        super().__init__()

    def fit(self, xs, ys):
        self.clf.fit(self.lda_xs(xs), ys)

    def predict(self, xs):
        return self.clf.predict(self.lda_xs(xs))

    def predict_proba(self, xs):
        return self.clf.predict_proba(self.lda_xs(xs))

    def lda_xs(self, xs):
        return self.lda.transform([x['text'] for x in xs])


def get_attributes(obj):
    if isinstance(obj, TfidfVectorizer):
        return get_tfidf_attributes(obj)
    return {attr: getattr(obj, attr) for attr in dir(obj)
            if not attr.startswith('_') and attr.endswith('_')}


def set_attributes(obj, attributes):
    if isinstance(obj, TfidfVectorizer):
        set_ifidf_attributes(obj, attributes)
    for k, v in attributes.items():
        setattr(obj, k, v)


def get_tfidf_attributes(obj):
    return {
        '_idf_diag': obj._tfidf._idf_diag,
        'vocabulary_': obj.vocabulary_,
    }


def set_ifidf_attributes(obj, attributes):
    obj._tfidf._idf_diag = attributes['_idf_diag']
    obj.vocabulary_ = attributes['vocabulary_']

from typing import Dict, Any

from eli5.sklearn.explain_weights import explain_weights
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier


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
        'xgboost': lambda : XGBClassifier(max_depth=2),
    }
    default_clf_kind = 'logcv'

    def __init__(self,
                 use_url=True,
                 use_text=True,
                 use_lda=False,
                 use_dmoz_fasttext=False,
                 use_dmoz_sklearn=False,
                 clf_kind=None):
        clf_kind = clf_kind or self.default_clf_kind
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
            self.lda = load_trained_model(
                'lda', lambda: joblib.load('dmoz-lda-limit10k.joblib'))
            # TODO - proper feature names
            vectorizers.append(
                ('lda', FunctionTransformer(
                    func=lambda xs: self.lda.transform(
                        [x['text'].lower() for x in xs]),
                    validate=False,
                )))
        if use_dmoz_fasttext or use_dmoz_sklearn:
            assert not (use_dmoz_fasttext and use_dmoz_sklearn)
            if use_dmoz_fasttext:
                import fasttext
                self.dmoz_clf = 'fasttext'
                self.dmoz_model = load_trained_model(
                    'dmoz_fasttext', lambda: fasttext.load_model(
                        'dmoz-ng1-mc10-mcl100.model.bin.bin'))
            else:
                import pickle
                self.dmoz_clf = 'sklearn'
                self.dmoz_model = load_trained_model(
                    'dmoz_sklearn', lambda: pickle.load(
                        open('dmoz_sklearn_full.pkl', 'rb')))
            # TODO - get rid of self.preprocess, do it in vectorizer
            self.dmoz_vec = PrefixDictVectorizer('dmoz')
            vectorizers.append(('dmoz', self.dmoz_vec))
        else:
            self.dmoz_vec = None
        if use_text:
            self.default_text_preprocessor = TfidfVectorizer().build_preprocessor()
            self.text_vec = TfidfVectorizer(preprocessor=self.text_preprocessor)
            vectorizers.append(('text', self.text_vec))
        else:
            self.text_vec = None
        self.vec = FeatureUnion(vectorizers)
        pipeline = [self.vec]
        if clf_kind == 'xgboost':
            # Work around xgboost issue:
            # https://github.com/dmlc/xgboost/issues/1238#issuecomment-243872543
            pipeline.append(CSCTransformer())
        self.clf = self.clf_kinds[clf_kind]()
        pipeline.append(self.clf)
        self.pipeline = make_pipeline(*pipeline)
        super().__init__(use_url=use_url,
                         use_text=use_text,
                         use_lda=use_lda,
                         use_dmoz_fasttext=use_dmoz_fasttext,
                         use_dmoz_sklearn=use_dmoz_sklearn,
                         clf_kind=clf_kind,
                         )

    def text_preprocessor(self, item):
        return self.default_text_preprocessor(item['text'])

    @staticmethod
    def url_preprocessor(item):
        return item['url'].lower()

    def preprocess(self, xs):
        if self.dmoz_vec:
            n_top = 10

            if self.dmoz_clf == 'fasttext':
                from hh_page_clf.pretraining.dmoz_fasttext import to_single_line
                for item, probs in zip(xs, self.dmoz_model.predict_proba([
                        to_single_line(x['text']) for x in xs], k=n_top)):
                    for label, prob in probs:
                        label = 'dmoz_{}'.format(label[len('__label__'):])
                        item[label] = 100 * prob

            elif self.dmoz_clf == 'sklearn':
                for item, probs in zip(
                        xs, self.dmoz_model['pipeline'].predict_proba(
                            [x['text'] for x in xs])):
                    label_probs = list(zip(self.dmoz_model['labels'], probs))
                    label_probs.sort(key=lambda x: x[1], reverse=True)
                    for label, prob in label_probs[:n_top]:
                        item['dmoz_{}'.format(label)] = 100 * prob

        return xs

    def fit(self, xs, ys):
        xs = self.preprocess(xs)
        if self.text_vec:
            vec = TfidfVectorizer(
                preprocessor=self.text_preprocessor,
                stop_words='english',
            )
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
        xs = self.preprocess(xs)
        return self.pipeline.predict(xs)

    def predict_proba(self, xs):
        xs = self.preprocess(xs)
        return self.pipeline.predict_proba(xs)

    def explain_weights(self):
        expl = explain_weights(self.clf, vec=self.vec, top=100)
        if expl.targets:
            fweights = expl.targets[0].feature_weights
            for fw_lst in [fweights.pos, fweights.neg]:
                for fw in fw_lst:
                    if fw.feature.startswith('text__'):
                        fw.feature = fw.feature[len('text__'):]
                    elif fw.feature.startswith('url__'):
                        fw.feature = 'url: {}'.format(fw.feature[len('url__'):])
                    elif fw.feature.startswith('dmoz__dmoz_'):
                        fw.feature = 'dmoz: {}'.format(fw.feature[len('dmoz__dmoz_'):])
        # TODO - same for feature importances
        return expl

    def get_params(self):
        return {
            'text_vec_attrs': get_attributes(self.text_vec),
            'url_vec_attrs': get_attributes(self.url_vec),
            'dmoz_vec_attrs': get_attributes(self.dmoz_vec),
            'clf_attrs': get_attributes(self.clf),
        }

    def set_params(self, *,
                   text_vec_attrs, url_vec_attrs, clf_attrs, dmoz_vec_attrs):
        set_attributes(self.text_vec, text_vec_attrs)
        set_attributes(self.url_vec, url_vec_attrs)
        set_attributes(self.dmoz_vec, dmoz_vec_attrs)
        set_attributes(self.clf, clf_attrs)


class PrefixDictVectorizer(DictVectorizer):
    def __init__(self, prefix, **kwargs):
        self.prefix = prefix
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        return super().fit(self.with_prefix(X), y=y)

    def fit_transform(self, X, y=None):
        return super().fit_transform(self.with_prefix(X), y=y)

    def transform(self, X, y=None):
        return super().transform(self.with_prefix(X), y=y)

    def with_prefix(self, xs):
        return [{k: v for k, v in item.items() if k.startswith(self.prefix)}
                for item in xs]


class CSCTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.tocsc()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


skip_attributes = {'feature_importances_'}


def get_attributes(obj):
    if isinstance(obj, TfidfVectorizer):
        return get_tfidf_attributes(obj)
    else:
        return {attr: getattr(obj, attr) for attr in dir(obj)
                if not attr.startswith('_') and attr.endswith('_')
                and attr not in skip_attributes}


def set_attributes(obj, attributes):
    if isinstance(obj, TfidfVectorizer):
        set_ifidf_attributes(obj, attributes)
    else:
        for k, v in attributes.items():
            try:
                setattr(obj, k, v)
            except AttributeError:
                raise AttributeError(
                    'can\'t set attribute {} on {}'.format(k, obj))


def get_tfidf_attributes(obj):
    return {
        '_idf_diag': obj._tfidf._idf_diag,
        'vocabulary_': obj.vocabulary_,
    }


def set_ifidf_attributes(obj, attributes):
    obj._tfidf._idf_diag = attributes['_idf_diag']
    obj.vocabulary_ = attributes['vocabulary_']


_trained_models_cache = {}


def load_trained_model(name, fn):
    if name not in _trained_models_cache:
        _trained_models_cache[name] = fn()
    return _trained_models_cache[name]

import pickle
from typing import Dict, Any

from eli5.sklearn.explain_weights import explain_weights
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.pipeline import make_pipeline, FeatureUnion
from scipy.sparse import issparse
from xgboost import XGBClassifier

from .utils import get_stop_words


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
        'extra_tree': lambda: ExtraTreesClassifier(
            n_estimators=100, random_state=42),
        'xgboost': lambda: XGBClassifier(max_depth=2, missing=0),
    }
    default_clf_kind = 'xgboost'

    def __init__(self,
                 use_url=True,
                 use_text=True,
                 lda=None,
                 doc2vec=None,
                 dmoz_fasttext=None,
                 dmoz_sklearn=None,
                 clf_kind=None):
        clf_kind = clf_kind or self.default_clf_kind
        vectorizers = []

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

        if lda:
            lda_model = load_trained_model('lda', load_joblib_or_pickle, lda)
            lda_feature_names = load_trained_model(
                'lda_feature_names', get_lda_feature_names, lda_model)
            vectorizers.append(
                ('lda', LDATransformer(lda_model, lda_feature_names)))

        if doc2vec:
            # This is experimental, not used by default.
            from gensim.models import Doc2Vec
            doc2vec_model = load_trained_model('doc2vec', Doc2Vec.load, doc2vec)
            vectorizers.append(('doc2vec', Doc2VecTransformer(doc2vec_model)))

        if dmoz_fasttext or dmoz_sklearn:
            # This is experimental, not used by default.
            assert not (dmoz_fasttext and dmoz_sklearn)
            if dmoz_fasttext:
                import fasttext
                self.dmoz_clf = 'fasttext'
                self.dmoz_model = load_trained_model(
                    'dmoz_fasttext', fasttext.load_model, dmoz_fasttext)
            else:
                self.dmoz_clf = 'sklearn'
                self.dmoz_model = load_trained_model(
                    'dmoz_sklearn',
                    lambda: pickle.load(open(dmoz_sklearn, 'rb')))
            # TODO - get rid of self.preprocess, do it in vectorizer
            self.dmoz_vec = PrefixDictVectorizer('dmoz')
            vectorizers.append(('dmoz', self.dmoz_vec))
        else:
            self.dmoz_vec = None

        if use_text:
            self.default_text_preprocessor = (
                CountVectorizer().build_preprocessor())
            self.text_vec = CountVectorizer(
                binary=True,
                preprocessor=self.text_preprocessor,
            )
            vectorizers.append(('text', self.text_vec))
        else:
            self.text_vec = None

        self.vec = FeatureUnion(vectorizers)
        self.clf = self.clf_kinds[clf_kind]()
        super().__init__(
            use_url=use_url,
            use_text=use_text,
            lda=lda,
            doc2vec=doc2vec,
            dmoz_fasttext=dmoz_fasttext,
            dmoz_sklearn=dmoz_sklearn,
            clf_kind=clf_kind,
        )

    @property
    def pipeline(self):
        # This is a property for serialization support with xgboost,
        # because we change self.clf after __init__.
        pipeline = [self.vec]
        if isinstance(self.clf, XGBClassifier):
            # Work around xgboost issue:
            # https://github.com/dmlc/xgboost/issues/1238#issuecomment-243872543
            pipeline.append(CSCTransformer())
        pipeline.append(self.clf)
        return make_pipeline(*pipeline)

    def text_preprocessor(self, item):
        return self.default_text_preprocessor(item['text'])

    @staticmethod
    def url_preprocessor(item):
        return item['url'].lower()

    def preprocess(self, xs):
        if self.dmoz_vec:
            n_top = 10
            if self.dmoz_clf == 'fasttext':
                self._add_dmoz_fasttext_features(xs, n_top)
            elif self.dmoz_clf == 'sklearn':
                self._add_dmoz_sklearn_features(xs, n_top)
        return xs

    def _add_dmoz_fasttext_features(self, xs, n_top):
        from hh_page_clf.pretraining.dmoz_fasttext import to_single_line
        for item, probs in zip(xs, self.dmoz_model.predict_proba(
                [to_single_line(x['text']) for x in xs], k=n_top)):
            for label, prob in probs:
                label = 'dmoz_{}'.format(label[len('__label__'):])
                item[label] = 100 * prob

    def _add_dmoz_sklearn_features(self, xs, n_top):
        for item, probs in zip(
                xs, self.dmoz_model['pipeline'].predict_proba(
                    [x['text'] for x in xs])):
            label_probs = list(zip(self.dmoz_model['labels'], probs))
            label_probs.sort(key=lambda x: x[1], reverse=True)
            for label, prob in label_probs[:n_top]:
                item['dmoz_{}'.format(label)] = 100 * prob

    def fit(self, xs, ys):
        xs = self.preprocess(xs)
        if self.text_vec:
            self.text_vec.vocabulary = self._select_text_features(xs, ys)
        self.pipeline.fit(xs, ys)

    def _select_text_features(self, xs, ys):
        vec = TfidfVectorizer(
            preprocessor=self.text_preprocessor,
            stop_words=get_stop_words(),
        )
        transformed = vec.fit_transform(xs)
        feature_selection_clf = SGDClassifier(
            loss='log', penalty='l2', n_iter=50, random_state=42)
        feature_selection_clf.fit(transformed, ys)
        abs_coefs = np.abs(feature_selection_clf.coef_[0])
        features = set((abs_coefs > np.mean(abs_coefs)).nonzero()[0])
        # FIXME - relies on ngram_range=(1, 1)
        return [w for w, idx in vec.vocabulary_.items() if idx in features]

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
                    fw.feature = self._prettify_feature(fw.feature)
        elif expl.feature_importances:
            importances = expl.feature_importances.importances
            for fw in importances:
                fw.feature = self._prettify_feature(fw.feature)
        return expl

    @staticmethod
    def _prettify_feature(feature):
        for prefix, tpl in [
            ('text__', '{}'),
            ('url__', 'URL: {}'),
            ('lda__', 'Topic: {}'),
            ('dmoz__dmoz_', 'Topic: {}'),
        ]:
            if feature.startswith(prefix):
                return tpl.format(feature[len(prefix):])
        return feature

    def get_params(self):
        return {
            'text_vec_attrs': get_attributes(self.text_vec),
            'url_vec_attrs': get_attributes(self.url_vec),
            'dmoz_vec_attrs': get_attributes(self.dmoz_vec),
            'clf_attrs': get_attributes(self.clf),
        }

    def set_params(self, *,
                   text_vec_attrs, url_vec_attrs, clf_attrs, dmoz_vec_attrs):
        set_attributes(self, 'text_vec', text_vec_attrs)
        set_attributes(self, 'url_vec', url_vec_attrs)
        set_attributes(self, 'dmoz_vec', dmoz_vec_attrs)
        set_attributes(self, 'clf', clf_attrs)


class PrefixDictVectorizer(DictVectorizer):
    def __init__(self, prefix, **kwargs):
        self.prefix = prefix
        super().__init__(**kwargs)

    def fit(self, xs, y=None):
        return super().fit(self.with_prefix(xs), y=y)

    def fit_transform(self, xs, y=None):
        return super().fit_transform(self.with_prefix(xs), y=y)

    def transform(self, xs, y=None):
        return super().transform(self.with_prefix(xs), y=y)

    def with_prefix(self, xs):
        return [{k: v for k, v in item.items() if k.startswith(self.prefix)}
                for item in xs]


class StatelessTransformer(TransformerMixin):
    def transform(self, xs, y=None, **fit_params):
        raise NotImplementedError

    def fit_transform(self, xs, y=None, **fit_params):
        self.fit(xs, y, **fit_params)
        return self.transform(xs)

    def fit(self, xs, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


class CSCTransformer(StatelessTransformer):
    def transform(self, xs, y=None, **fit_params):
        return xs.tocsc() if issparse(xs) else xs


class LDATransformer(StatelessTransformer):
    def __init__(self, lda_pipeline, feature_names):
        self.lda = lda_pipeline
        self.feature_names = feature_names
        super().__init__()

    def transform(self, xs, y=None, **fit_params):
        return self.lda.transform([x['text'].lower() for x in xs])

    def get_feature_names(self):
        return self.feature_names


def get_lda_feature_names(lda_pipeline):
    (_, vec), (_, lda) = lda_pipeline.steps
    vec_feature_names = vec.get_feature_names()
    return [', '.join(vec_feature_names[idx] for idx in topic_indices)
            for topic_indices in top_indices(lda, top=3)]


def top_indices(lda, top):
    dim = lda.components_.shape[0]
    indices = np.argpartition(lda.components_, -top, axis=1)[:, -top:]
    row_indices = np.tile(np.arange(dim), [top, 1]).T
    values = lda.components_[row_indices, indices]
    return indices[row_indices, np.argsort(-values)]


class Doc2VecTransformer(StatelessTransformer):
    def __init__(self, doc2vec):
        from gensim.models import Doc2Vec
        self.doc2vec = doc2vec  # type: Doc2Vec
        super().__init__()

    def transform(self, xs, y=None, **fit_params):
        from hh_page_clf.pretraining.train_doc2vec import tokenize
        return np.array([self.doc2vec.infer_vector(tokenize(x['text']))
                         for x in xs])

    def get_feature_names(self):
        n_dim = self.doc2vec.vector_size
        return [str(i + 1) for i in range(n_dim)]


skip_attributes = {'feature_importances_'}


def get_attributes(obj):
    if isinstance(obj, TfidfVectorizer):
        return get_tfidf_attributes(obj)
    elif isinstance(obj, XGBClassifier):
        return pickle.dumps(obj)
    elif isinstance(obj, BaseEstimator):
        return {attr: getattr(obj, attr) for attr in dir(obj)
                if not attr.startswith('_') and attr.endswith('_')
                and attr not in skip_attributes}
    elif obj is not None:
        raise TypeError(type(obj))


def set_attributes(parent, field, attributes):
    obj = getattr(parent, field)
    if isinstance(obj, TfidfVectorizer):
        set_ifidf_attributes(obj, attributes)
    elif isinstance(obj, XGBClassifier):
        setattr(parent, field, pickle.loads(attributes))
    elif isinstance(obj, BaseEstimator):
        for k, v in attributes.items():
            try:
                setattr(obj, k, v)
            except AttributeError:
                raise AttributeError(
                    'can\'t set attribute {} on {}'.format(k, obj))
    elif obj is not None:
        raise TypeError(type(obj))


def get_tfidf_attributes(obj):
    return {
        '_idf_diag': obj._tfidf._idf_diag,
        'vocabulary_': obj.vocabulary_,
    }


def set_ifidf_attributes(obj, attributes):
    obj._tfidf._idf_diag = attributes['_idf_diag']
    obj.vocabulary_ = attributes['vocabulary_']


_trained_models_cache = {}


def load_trained_model(name, fn, *args, **kwargs):
    if name not in _trained_models_cache:
        _trained_models_cache[name] = fn(*args, **kwargs)
    return _trained_models_cache[name]


def load_joblib_or_pickle(filename):
    if filename.endswith('joblib'):
        return joblib.load(filename)
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)

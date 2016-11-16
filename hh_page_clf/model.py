import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.pipeline import Pipeline, make_union


def default_fit_clf(xs, ys) -> Pipeline:
    # Doing explicit feature selection because:
    # - eli5 does not support pipelines with feature selection
    # - final model should be small and does not need the whole vocabulary
    default_text_preprocessor = TfidfVectorizer().build_preprocessor()
    text_preprocessor = lambda x: default_text_preprocessor(x['text'])
    vec = TfidfVectorizer(preprocessor=text_preprocessor)
    transformed = vec.fit_transform(xs)
    feature_selection_clf = SGDClassifier(
        loss='log', penalty='l2', n_iter=50, random_state=42)
    feature_selection_clf.fit(transformed, ys)
    abs_coefs = np.abs(feature_selection_clf.coef_[0])
    features = set((abs_coefs > np.mean(abs_coefs)).nonzero()[0])
    # FIXME - relies on ngram_range=(1, 1)
    vocabulary = [w for w, idx in vec.vocabulary_.items() if idx in features]
    clf = Pipeline([
        ('vec', make_union(
            TfidfVectorizer(
                vocabulary=vocabulary,
                preprocessor=text_preprocessor,
            ),
            CountVectorizer(
                binary=True,
                analyzer='char',
                ngram_range=(3, 4),
                preprocessor=lambda x: x['url'].lower(),
            ),
        )),
        ('clf', LogisticRegressionCV(random_state=42)),
    ])
    clf.fit(xs, ys)
    return clf

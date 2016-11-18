import re
import tempfile

import numpy as np
import fasttext

from .model import BaseModel


class FastTextModel(BaseModel):
    def fit(self, xs, ys):
        with tempfile.NamedTemporaryFile(mode='wt', delete=False) as f:
            self.f = f
            for x, y in zip(xs, ys):
                f.write('__label__{label} {text}\n'.format(
                    label=int(y),
                    text=self._text(x),
                ))
        self.clf = fasttext.supervised(f.name, 'model')

    def _text(self, x):
        return '{url} {text}\n'.format(
            url=x['url'].lower(),
            text=x['text'].replace('\n', ' '),
           #text=' '.join(re.findall('\w+', x['text'].replace('\n', ' ').lower())),
        )

    def predict(self, xs):
        ys = self.clf.predict([self._text(x) for x in xs])
        return np.array([False if not y else y[0] == '1' for y in ys])

    def predict_proba(self, xs):
        ys = self.clf.predict_proba([self._text(x) for x in xs])
        # FIXME - judging by ROC AUC, something is wrong here
        return np.array([(1 - prob, prob) for prob in map(self._get_prob, ys)])

    def _get_prob(self, y):
        if not y:
            return 0
        (class_, prob), = y
        assert class_ in '01'
        return prob if class_ == '1' else (1 - prob)

from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline

from hh_page_clf.train import train_model


def test_train_model():
    data = fetch_20newsgroups(
        random_state=42,
        categories=['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'])
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
Metrics:
Accuracy            :   0.983 ± 0.008
F1                  :   0.973 ± 0.011
ROC AUC             :   nan ± nan
Dataset: 2373 documents, 75% with labels
Class balance: 33% relevant, 67% not relevant
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

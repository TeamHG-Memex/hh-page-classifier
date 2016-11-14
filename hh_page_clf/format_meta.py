import re

from eli5.formatters import format_as_html, fields
from jinja2 import Environment, PackageLoader

from .train import Meta, ERROR, WARNING, NOTICE


template_env = Environment(
    loader=PackageLoader('hh_page_clf', 'templates'),
    extensions=['jinja2.ext.with_'])
template_env.filters.update(dict(
    add_tooltips=lambda x: _add_tooltips(x),
    alert_class=lambda x: _alert_class(x),
))


def format_meta(meta: Meta) -> str:
    """ Format meta as html (to be moved to THH).
    """
    template = template_env.get_template('meta.html')
    weights = None
    if meta.weights_explanation:
        weights = format_as_html(meta.weights_explanation, show=fields.WEIGHTS)
        weights = re.sub('<p>.*</p>', '', weights, flags=re.S)  # FIXME
    return template.render(meta=meta, weights=weights)


TOOLTIPS = {
    'ROC AUC': (
        'Area under ROC (receiver operating characteristic) curve '
        'shows how good is the classifier at telling relevant pages from '
        'non-relevant at different thresholds. '
        'Random classifier has ROC&nbsp;AUC&nbsp;=&nbsp;0.5, '
        'and a perfect classifier has ROC&nbsp;AUC&nbsp;=&nbsp;1.0.'
    ),
    'Accuracy': (
        'Accuracy is the ratio of pages classified correctly as '
        'relevant or not relevant. This metric is easy to interpret but '
        'not very good for unbalanced datasets.'
    ),
    'F1': (
        'F1 score is a combination of recall and precision for detecting '
        'relevant pages. It shows how good is a classifier at detecting '
        'relevant pages at default threshold.'
        'Worst value is 0.0 and perfect value is 1.0.'
    ),
}


def _add_tooltips(text: str) -> str:
    for term, explanation in TOOLTIPS.items():
        tooltip = (
            '{term}'
            '<span data-toggle="tooltip"'
            ' data-placement="bottom"'
            ' title="{explanation}">'
            '<span class="question glyphicon glyphicon-question-sign"></span>'
            '</span>'
        ).format(explanation=explanation, term=term)
        text = text.replace(term, tooltip)
    return text


def _alert_class(heading: str) -> str:
    return {
        ERROR: 'alert-danger',
        WARNING: 'alert-warning',
        NOTICE: 'alert-success',
    }[heading]

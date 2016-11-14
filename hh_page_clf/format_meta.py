from eli5.formatters import format_as_html, fields
from jinja2 import Environment, PackageLoader

from .train import Meta


template_env = Environment(
    loader=PackageLoader('hh_page_clf', 'templates'),
    extensions=['jinja2.ext.with_'])


def format_meta(meta: Meta) -> str:
    """ Format meta as html (to be moved to THH).
    """
    template = template_env.get_template('meta.html')
    return template.render(
        meta=meta,
        weights=format_as_html(meta.weights_explanation, show=fields.WEIGHTS)
        if meta.weights_explanation else None,
    )

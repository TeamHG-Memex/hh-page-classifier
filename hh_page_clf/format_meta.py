from jinja2 import Environment, PackageLoader

from .train import Meta, ERROR, WARNING, NOTICE, TOOLTIPS


# This is all just for testing, will be removed.
# Real html formatting is done in the THH.


template_env = Environment(
    loader=PackageLoader('hh_page_clf', 'templates'),
    extensions=['jinja2.ext.with_'])
template_env.filters.update(dict(
    add_tooltips=lambda x: _add_tooltips(x),
    alert_class=lambda x: _alert_class(x),
))


def format_meta(meta: Meta) -> str:
    template = template_env.get_template('meta.html')
    return template.render(meta=meta)


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

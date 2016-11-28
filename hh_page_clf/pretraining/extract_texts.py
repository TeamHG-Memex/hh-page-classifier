import argparse
import gzip
from functools import partial
try:
    import ujson as json
except ImportError:
    import json
import multiprocessing

import json_lines
import html_text


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('input', help='In .jl.gz format with html in "text" field')
    arg('html_field', help='Field name where html is stored')
    arg('output', help='Output in jl.gz format with text in "text" field')
    args = parser.parse_args()

    with json_lines.open(args.input, broken=True) as f, gzip.open(
            args.output, 'wt') as outf:
        with multiprocessing.Pool() as pool:
            for text_item in pool.imap_unordered(
                    partial(text_worker, html_field=args.html_field), f):
                outf.write(json.dumps(text_item))
                outf.write('\n')


def text_worker(item, html_field):
    url = item.get('url')
    html = item.get(html_field)
    try:
        text = html_text.extract_text(html)
    except UnicodeEncodeError:
        text = html
    text_item = {'text': text}
    if url is not None:
        text_item['url'] = url
    return text_item

import argparse
import gzip
import csv
import re

import json_lines


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('dmoz_urls_topics', help='In .csv.gz format')
    arg('dmoz_text', help='Items with url and text fields in .jl.gz format')
    arg('output', help='output file for fasttext training')
    args = parser.parse_args()

    with gzip.open(args.dmoz_urls_topics, 'rt') as f:
        topics_by_url = dict(csv.reader(f))

    with json_lines.open(args.dmoz_text) as f, open(args.output, 'wt') as outf:
        for item in f:
            topics = topics_by_url[item['url']]
            topics = topics.split('/')
            if topics[0] == 'Top':
                topics = topics[1:]
            for topic in topics:
                outf.write('__label__{} '.format(topic))
            outf.write(to_single_line(item['text']))
            outf.write('\n')


def to_single_line(text):
    return re.sub('\s+', ' ', text.replace('\n', ' '))

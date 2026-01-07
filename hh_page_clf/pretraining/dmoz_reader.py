import argparse
import csv
import gzip
from xml.dom import pulldom


def dmoz_reader(filename):
    doc = pulldom.parse(filename)
    for event, node in doc:
        if event == pulldom.START_ELEMENT and node.tagName == 'ExternalPage':
            doc.expandNode(node)
            url = node.attributes['about'].value
            topic_node = node.getElementsByTagName('topic')[0]
            topics = topic_node.childNodes[0].data
            yield url, topics


def to_csv():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('input', help='path to dmoz data in RDF format (content.rdf.u8)')
    arg('output', help='will be written in .csv.gz format')
    args = parser.parse_args()
    with gzip.open(args.output, 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'topics'])
        for url, topics in dmoz_reader(args.input):
            writer.writerow([url, topics])

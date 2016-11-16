import argparse
import gzip
import hashlib
import logging
import json
from pprint import pformat
from typing import Dict, List, Tuple

import attr
from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord

from .train import train_model, AdviceItem, ERROR, Meta, ModelMeta
from .utils import configure_logging


class Service:
    input_topic = 'dd-modeler-input'
    output_topic = 'dd-modeler-output'
    max_message_size = 104857600

    def __init__(self, kafka_host=None, fit_clf=None, debug=False):
        self.fit_clf = fit_clf
        kafka_kwargs = {}
        if kafka_host is not None:
            kafka_kwargs['bootstrap_servers'] = kafka_host
        self.consumer = KafkaConsumer(
            self.input_topic,
            max_partition_fetch_bytes=self.max_message_size,
            consumer_timeout_ms=10,
            **kafka_kwargs)
        self.producer = KafkaProducer(
            max_request_size=self.max_message_size,
            **kafka_kwargs)
        self.debug = debug
        self.stop_marker = object()

    def run(self) -> None:
        """ Listen to messages with data to train on, and return trained models
        with a report on model quality.
        If several messages with the same id arrive, result of only the last one
        will be sent back.
        """
        to_send = []  # type: List[Tuple[str, Dict]]
        while True:
            requests = {}  # type: Dict[str, ConsumerRecord]
            order = {}  # type: Dict[str, int]
            for idx, message in enumerate(self.consumer):
                value = self.extract_value(message)
                if value is self.stop_marker:
                    return
                elif value is not None:
                    id_ = value['id']
                    requests[id_] = value
                    order[id_] = idx
            self.consumer.commit()
            for id_, result in to_send:
                if id_ in requests:
                    logging.info(
                        'Dropping result for id "{}", as new request arrived'
                        .format(id_))
                else:
                    self.send_result(result)
            # Ordering is important only to simplify testing.
            to_send = [(id_, self.train_model(request))
                       for id_, request in sorted(requests.items(),
                                                  key=lambda x: order[x[0]])]

    def extract_value(self, message):
        self._debug_save_message(message.value, 'incoming')
        try:
            value = json.loads(message.value.decode('utf8'))
        except Exception as e:
            logging.error('Error decoding message: {}'
                          .format(repr(message.value)),
                          exc_info=e)
            return
        if value == {'from-tests': 'stop'}:
            logging.info('Got message to stop (from tests)')
            return self.stop_marker
        elif isinstance(value.get('pages'), list) and value.get('id'):
            logging.info(
                'Got training task with {pages} pages, id "{id}", '
                'message checksum {checksum}, offset {offset}.'
                .format(
                    pages=len(value['pages']),
                    id=value.get('id'),
                    checksum=message.checksum,
                    offset=message.offset,
                ))
            return value
        else:
            logging.error(
                'Dropping a message without "pages" or "id" key: {}'
                .format(pformat(value)))

    def train_model(self, request: Dict) -> Dict:
        try:
            result = train_model(request['pages'], fit_clf=self.fit_clf)
        except Exception as e:
            logging.error('Failed to train a model', exc_info=e)
            result = ModelMeta(
                model=None,
                meta=Meta(advice=[AdviceItem(
                    ERROR,
                    'Unknown error while training a model: {}'.format(e))]))
        return {
            'id': request['id'],
            'quality': json.dumps(attr.asdict(result.meta)),
            'model': (result.model.encode() if result.model is not None
                      else None),
        }

    def send_result(self, result: Dict) -> None:
        message = json.dumps(result).encode('utf8')
        self._debug_save_message(message, 'outgoing')
        logging.info('Sending result for id "{}", model size {:,} bytes'
                     .format(result.get('id'),
                             len(result.get('model') or '')))
        self.producer.send(self.output_topic, message)
        self.producer.flush()

    def _debug_save_message(self, message: bytes, kind: str) -> None:
        if self.debug:
            filename = ('hh-page-clf-{}.json.gz'
                        .format(hashlib.md5(message).hexdigest()))
            logging.info('Saving {} message to {}'.format(kind, filename))
            with gzip.open(filename, 'wb') as f:
                f.write(message)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--kafka-host')
    arg('--debug', action='store_true')
    args = parser.parse_args()

    configure_logging()
    service = Service(kafka_host=args.kafka_host, debug=args.debug)
    logging.info('Starting hh page classifier service')
    service.run()

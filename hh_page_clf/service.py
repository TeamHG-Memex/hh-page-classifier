import argparse
import base64
import logging
import json
import pickle
from pprint import pformat
from typing import Dict, List, Tuple, Optional

from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord

from .train import train_model
from .utils import configure_logging


class Service:
    input_topic = 'dd-modeler-input'
    output_topic = 'dd-modeler-output'

    def __init__(self, kafka_host=None, init_clf=None):
        self.init_clf = init_clf
        kafka_kwargs = {}
        if kafka_host is not None:
            kafka_kwargs['bootstrap_servers'] = kafka_host
        self.consumer = KafkaConsumer(
            self.input_topic,
            consumer_timeout_ms=10,
            **kafka_kwargs)
        self.producer = KafkaProducer(
            value_serializer=encode_message,
            **kafka_kwargs)
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
                else:
                    logging.error(
                        'Dropping a message without "pages" or "id" key: {}'
                        .format(pformat(value)))
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
            result = train_model(request['pages'], init_clf=self.init_clf)
        except Exception as e:
            logging.error('Failed to train a model', exc_info=e)
            return {
                'id': request['id'],
                'quality': 'Unknown error while training a model: {}'.format(e),
                'model': None,
            }
        else:
            return {
                'id': request['id'],
                'quality': result.meta,
                'model': encode_model(result.model),
            }

    def send_result(self, result: Dict) -> None:
        logging.info('Sending result for id "{}", model size {} bytes'
                     .format(result.get('id'),
                             len(result.get('model') or '')))
        self.producer.send(self.output_topic, result)
        self.producer.flush()


def encode_message(message: Dict) -> bytes:
    try:
        return json.dumps(message).encode('utf8')
    except Exception as e:
        logging.error('Error serializing message', exc_info=e)
        raise


def encode_model(model: object) -> Optional[str]:
    if model is not None:
        return base64.b64encode(pickle.dumps(model, protocol=2)).decode('ascii')


def decode_model(data: Optional[str]) -> object:
    if data is not None:
        return pickle.loads(base64.b64decode(data))


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--kafka-host')
    args = parser.parse_args()

    configure_logging()
    service = Service(kafka_host=args.kafka_host)
    logging.info('Starting hh page classifier service')
    service.run()

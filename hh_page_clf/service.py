import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
import gzip
import hashlib
from functools import partial
import logging
import json
from pprint import pformat
from typing import Dict, Optional, Tuple

import attr
from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord

from .train import train_model, AdviceItem, ERROR, Meta, ModelMeta
from .utils import configure_logging, encode_object


class Service:
    input_topic = 'dd-modeler-input'
    output_topic = 'dd-modeler-output'
    progress_output_topic = 'dd-modeler-progress'
    max_message_size = 104857600

    def __init__(self, kafka_host=None, model_cls=None, model_kwargs=None,
                 debug=False):
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs or {}
        kafka_kwargs = {}
        if kafka_host is not None:
            kafka_kwargs['bootstrap_servers'] = kafka_host
        self.consumer = KafkaConsumer(
            self.input_topic,
            group_id='{}-group'.format(self.input_topic),
            max_partition_fetch_bytes=self.max_message_size,
            consumer_timeout_ms=100,
            **kafka_kwargs)
        self.producer = KafkaProducer(
            max_request_size=self.max_message_size,
            **kafka_kwargs)
        self.debug = debug

    def run_loop(self) -> None:
        """ Listen to messages with data to train on, and return trained models
        with a report on model quality.
        If several messages with the same id arrive, result of only the last one
        will be sent back.
        This method loops until a message to stop is received (sent only from tests).
        """
        jobs = OrderedDict()  # type: Dict[str, Future]
        with ThreadPoolExecutor(max_workers=1) as pool:
            while True:
                to_submit = {}
                for message in self.consumer:
                    value, should_stop = self.extract_value(message)
                    if should_stop:
                        return
                    elif value is not None:
                        to_submit[value['id']] = value
                for id_, value in to_submit.items():
                    if id_ in jobs:
                        jobs[id_].cancel()
                    jobs[id_] = pool.submit(self.train_model, value)
                self.consumer.commit()
                sent = []
                for id_, future in jobs.items():
                    try:
                        result = future.result(timeout=0)
                    except TimeoutError:
                        pass
                    else:
                        self.send_result(result)
                        sent.append(id_)
                for id_ in sent:
                    del jobs[id_]

    def extract_value(self, message: ConsumerRecord
                      ) -> Tuple[Optional[Dict], bool]:
        self._debug_save_message(message.value, 'incoming')
        try:
            value = json.loads(message.value.decode('utf8'))
        except Exception as e:
            logging.error('Error decoding message: {}...'
                          .format(repr(message.value)[:100]),
                          exc_info=e)
            return None, False
        if value == {'from-tests': 'stop'}:
            logging.info('Got message to stop (from tests)')
            return None, True
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
            return value, False
        else:
            logging.error(
                'Dropping a message without "pages" or "id" key: {}'
                .format(pformat(value)))
            return None, False

    def train_model(self, request: Dict) -> Dict:
        id_ = request['id']
        try:
            result = train_model(
                request['pages'], model_cls=self.model_cls,
                progress_callback=partial(self.progress_callback, id_=id_),
                **self.model_kwargs)
        except Exception as e:
            logging.error('Failed to train a model', exc_info=e)
            result = ModelMeta(
                model=None,
                meta=Meta(advice=[AdviceItem(
                    ERROR,
                    'Unknown error while training a model: {}'.format(e))]))
        return {
            'id': id_,
            'quality': json.dumps(attr.asdict(result.meta)),
            'model': (encode_object(result.model) if result.model is not None
                      else None),
        }

    def progress_callback(self, progress: float, id_: str):
        logging.info('Sending progress update for {}: {:.0%}'
                     .format(id_, progress))
        self.producer.send(
            self.progress_output_topic, _encode_message({
                'id': id_,
                'percentage_done': 100 * progress,
            })
        )
        self.producer.flush()

    def send_result(self, result: Dict) -> None:
        message = _encode_message(result)
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


def _encode_message(value: Dict) -> bytes:
    return json.dumps(value).encode('utf8')


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--kafka-host')
    arg('--debug', action='store_true')
    arg('--random-pages', help='path to random negative pages (.jl.gz)')
    arg('--lda', help='path to LDA model (LDA is not used by default)')
    args = parser.parse_args()

    configure_logging()
    service = Service(
        kafka_host=args.kafka_host, debug=args.debug,
        model_kwargs=dict(
            random_pages=args.random_pages,
            lda=args.lda,
        ))
    logging.info('Starting hh page classifier service')
    service.run_loop()

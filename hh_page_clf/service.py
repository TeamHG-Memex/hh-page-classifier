import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
from functools import lru_cache
import gzip
import hashlib
from functools import partial
import logging
import json
from pprint import pformat
import requests
from typing import Dict, Optional, List, Tuple

import attr
from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord

from .train import train_model, AdviceItem, ERROR, Meta, ModelMeta
from .utils import configure_logging, encode_object


class Service:
    input_topic = 'dd-modeler-input'
    output_topic = 'dd-modeler-output'
    trainer_topic = 'dd-trainer-input'
    progress_output_topic = 'dd-modeler-progress'
    group_id = 'hh-page-classifier'
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
            group_id=self.group_id,
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
        This method loops until a message to stop is received
        (sent only from tests).
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
                        to_submit[value['workspace_id']] = value
                for ws_id, value in to_submit.items():
                    if ws_id in jobs:
                        _, future = jobs[ws_id]
                        future.cancel()
                    jobs[ws_id] = (value, pool.submit(self.train_model, value))
                sent = []
                for ws_id, (request, future) in jobs.items():
                    try:
                        result = future.result(timeout=0)
                    except TimeoutError:
                        pass
                    else:
                        self.send_result(result, request)
                        sent.append(ws_id)
                for ws_id in sent:
                    del jobs[ws_id]

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
        elif isinstance(value.get('pages'), list) and value.get('workspace_id'):
            logging.info(
                'Got training task with {pages} pages, workspace_id "{ws_id}"'
                .format(pages=len(value['pages']), ws_id=value['workspace_id']))
            return value, False
        else:
            logging.error(
                'Dropping a message without "pages" or "workspace_id" key: {}'
                .format(pformat(value)))
            return None, False

    def train_model(self, request: Dict) -> Dict:
        ws_id = request['workspace_id']
        try:
            pages = request['pages']
            pages = self._fetch_pages_html(pages)
            result = train_model(
                pages, model_cls=self.model_cls,
                progress_callback=partial(self.progress_callback, ws_id=ws_id),
                **self.model_kwargs)
        except Exception as e:
            logging.error('Failed to train a model', exc_info=e)
            result = ModelMeta(
                model=None,
                meta=Meta(advice=[AdviceItem(
                    ERROR,
                    'Unknown error while training a model: {}'.format(e))]))
        return {
            'workspace_id': ws_id,
            'quality': json.dumps(attr.asdict(result.meta)),
            'model': (encode_object(result.model) if result.model is not None
                      else None),
        }

    def _fetch_pages_html(self, pages: List[Dict]) -> List[Dict]:
        fetched = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for html, page in zip(
                    pool.map(_fetch_html,
                             [p.pop('html_location') for p in pages]),
                    pages):
                if html is not None:
                    page['html'] = html
                    fetched.append(page)
        return fetched

    def progress_callback(self, progress: float, ws_id: str):
        logging.info('Sending progress update for {}: {:.0%}'
                     .format(ws_id, progress))
        self.producer.send(
            self.progress_output_topic, _encode_message({
                'workspace_id': ws_id,
                'percentage_done': 100 * progress,
            })
        )
        self.producer.flush()

    def send_result(self, result: Dict, request: Dict) -> None:
        message = _encode_message(result)
        self._debug_save_message(message, 'outgoing')
        logging.info('Sending result for workspace "{}", model size {:,} bytes'
                     .format(result['workspace_id'],
                             len(result.get('model') or '')))
        self.producer.send(self.output_topic, message)
        if result.get('model'):
            self.producer.send(self.trainer_topic, _encode_message({
                'workspace_id': result['workspace_id'],
                'urls': [page['url'] for page in request['pages']
                         if page.get('relevant')],
                'page_model': result['model'],
            }))
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


@lru_cache(maxsize=1000)
def _fetch_html(html_location: str) -> str:
    if html_location.startswith('html://'):  # used in tests
        return html_location[len('html://'):]
    else:
        try:
            data = requests.get(html_location).json()
            return data['_source']['result']['crawlResultDto']['html']
        except Exception as e:
            logging.warning('Error fetching html for "{}": {}'
                            .format(html_location, e))


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

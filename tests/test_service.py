from collections import namedtuple
import json
import logging
from pprint import pprint
import threading
from typing import Dict

from html_text import extract_text
from kafka import KafkaConsumer, KafkaProducer

from hh_page_clf.models import BaseModel
from hh_page_clf.service import Service
from hh_page_clf.utils import configure_logging, decode_object, encode_object
from .test_train import ATestModel


configure_logging()


class ATestService(Service):
    input_topic = 'test-{}'.format(Service.input_topic)
    ouput_topic = 'test-{}'.format(Service.output_topic)
    trainer_topic = 'test-{}'.format(Service.trainer_topic)
    progress_output_topic = 'test-{}'.format(Service.progress_output_topic)
    group_id = None  # this massively simplifies testing


def encode_message(message: Dict) -> bytes:
    try:
        return json.dumps(message).encode('utf8')
    except Exception as e:
        logging.error('Error serializing message', exc_info=e)
        raise


def test_training():
    producer = KafkaProducer(value_serializer=encode_message)
    consumer = KafkaConsumer(ATestService.output_topic,
                             value_deserializer=decode_message)
    trainer_consumer = KafkaConsumer(ATestService.trainer_topic,
                                     value_deserializer=decode_message)
    for c in [consumer, trainer_consumer]:
        c.poll(timeout_ms=10)
    service = ATestService(model_cls=ATestModel, debug=False)
    service_thread = threading.Thread(target=service.run_loop)
    service_thread.start()
    pages = [{'url': 'http://example-{}.com/foo'.format(n),
              'html': '<h1>hi example-{} number{}</h1>'.format(n, n % 3),
              'relevant': n % 3 == 1,
            } for n in range(100)]
    train_request = {
        'pages': [{'url': page['url'],
                   'html_location': 'html://{}'.format(page['html']),
                   'relevant': page['relevant']}
                  for page in pages] +
                 [{'url': 'http://example.com',
                   'html_location': 'bad',
                   'relevant': True}]
    }

    def _test(train_response):
        model = decode_object(train_response.pop('model'))  # type: BaseModel
        pprint(train_response)
        pprint(json.loads(train_response['quality']))

        page_neg, page_pos = pages[:2]
        pred_proba = lambda page: \
            model.predict_proba([{'text': extract_text(page['html'])}])[0][1]
        assert pred_proba(page_pos) > 0.5
        assert pred_proba(page_neg) < 0.5
        return train_response

    try:
        request_1 = dict(train_request, workspace_id='some id 1')
        request_2 = dict(train_request, workspace_id='some id 2')
        producer.send(ATestService.input_topic, request_1)
        producer.send(ATestService.input_topic, request_1)
        producer.send(ATestService.input_topic, request_2)
        producer.flush()
        responses = get_responses(consumer)
        for r in responses:
            _test(r)
        assert ({r['workspace_id'] for r in responses} ==
                {'some id 1', 'some id 2'})
        producer.send(ATestService.input_topic, request_1)
        producer.flush()
        producer.send(ATestService.input_topic, request_1)
        producer.send(ATestService.input_topic, {'junk': True})
        producer.send(ATestService.input_topic, request_2)
        producer.send(ATestService.input_topic,
                      {'workspace_id': '3', 'pages': [True]})
        producer.flush()
        responses = get_responses(consumer)
        assert ({r['workspace_id'] for r in responses
                 if r['workspace_id'] != '3'} == {'some id 1', 'some id 2'})
        error_response = [r for r in responses if r['workspace_id'] == '3'][0]
        assert 'Error' in error_response['quality']
        assert "'bool' object" in error_response['quality']
        trainer_responses = get_responses(trainer_consumer, timeout_ms=10)
        assert len(trainer_responses) == 4
        assert all(r.get(k) for r in trainer_responses
                   for k in ['workspace_id', 'page_model', 'urls'])
    finally:
        producer.send(ATestService.input_topic, {'from-tests': 'stop'})
        producer.flush()
        service_thread.join()


def get_responses(consumer: KafkaConsumer, timeout_ms=2000):
    values = []
    while True:
        new_values = [
            r.value for v in consumer.poll(timeout_ms=timeout_ms).values()
            for r in v]
        if values and not new_values:
            break
        values.extend(new_values)
    return values


Point = namedtuple('Point', 'x, y')


def test_encode_object():
    p = Point(-1, 2.25)
    assert isinstance(encode_object(p), str)
    assert p == decode_object(encode_object(p))
    assert decode_object(None) is None
    assert encode_object(None) is None


def decode_message(message: bytes) -> Dict:
    try:
        return json.loads(message.decode('utf8'))
    except Exception as e:
        logging.error('Error deserializing message', exc_info=e)
        raise

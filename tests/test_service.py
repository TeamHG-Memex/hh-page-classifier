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


def clear_topics():
    for topic in [ATestService.input_topic, ATestService.output_topic]:
        consumer = KafkaConsumer(topic, consumer_timeout_ms=100)
        for _ in consumer:
            pass
        consumer.commit()


def encode_message(message: Dict) -> bytes:
    try:
        return json.dumps(message).encode('utf8')
    except Exception as e:
        logging.error('Error serializing message', exc_info=e)
        raise


def test_training():
    clear_topics()
    producer = KafkaProducer(value_serializer=encode_message)
    consumer = KafkaConsumer(
        ATestService.output_topic,
        value_deserializer=decode_message)
    service = ATestService(model_cls=ATestModel, debug=False)
    service_thread = threading.Thread(target=service.run)
    service_thread.start()
    train_request = {
        'pages': [
            {
                'url': 'http://example-{}.com/foo'.format(n),
                'html': '<h1>hi example-{} number{}</h1>'.format(n, n % 3),
                'relevant': n % 3 == 1,
            }
            for n in range(10)
        ]
    }

    def _test(request):
        train_response = next(consumer).value
        model = decode_object(train_response.pop('model'))  # type: BaseModel
        pprint(train_response)
        pprint(json.loads(train_response['quality']))

        assert train_response['id'] == request['id']

        page_neg, page_pos = request['pages'][:2]
        pred_proba = lambda page: \
            model.predict_proba([{'text': extract_text(page['html'])}])[0][1]
        assert pred_proba(page_pos) > 0.5
        assert pred_proba(page_neg) < 0.5

    try:
        request_1 = dict(train_request, id='some id 1')
        request_2 = dict(train_request, id='some id 2')
        producer.send(ATestService.input_topic, request_1)
        producer.send(ATestService.input_topic, request_1)
        producer.send(ATestService.input_topic, request_2)
        producer.flush()
        _test(request_1)
        _test(request_2)
        producer.send(ATestService.input_topic, request_1)
        producer.flush()
        producer.send(ATestService.input_topic, request_1)
        producer.send(ATestService.input_topic, {'junk': True})
        producer.send(ATestService.input_topic, request_2)
        producer.send(ATestService.input_topic, {'id': '3', 'pages': [True]})
        producer.flush()
        _test(request_1)
        _test(request_2)
        error_response = next(consumer).value
        assert error_response['id'] == '3'
        assert 'Error' in error_response['quality']
        assert "'bool' object has no attribute 'get'" in error_response['quality']
    finally:
        producer.send(ATestService.input_topic, {'from-tests': 'stop'})
        producer.flush()
        service_thread.join()


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

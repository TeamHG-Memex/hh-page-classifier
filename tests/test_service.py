from collections import namedtuple
import json
import logging
from pprint import pprint
import threading
from typing import Dict

from html_text import extract_text
from kafka import KafkaConsumer, KafkaProducer
from sklearn.pipeline import Pipeline

from hh_page_clf.service import Service, encode_model, decode_model
from hh_page_clf.utils import configure_logging
from .test_train import fit_clf


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
    service = ATestService(fit_clf=fit_clf, debug=False)
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
        model = decode_model(train_response.pop('model'))  # type: Pipeline
        pprint(train_response)
        pprint(json.loads(train_response['quality']))

        assert train_response == {
            'id': request['id'],
            'quality': json.dumps([
                ('Warning',
                 'Number of labeled documents is just 10, consider having at least 100 '
                 'labeled.'),
                ('Dataset',
                 '10 documents, 10 with labels (100%) across 10 domains.'),
                ('Class balance', '30% relevant, 70% not relevant.'),
                ('Metrics', ''),
                ('Accuracy', '1.000 ± 0.000'),
                ('F1', '0.750 ± 0.849'),
                ('ROC AUC', 'nan ± nan'),
                ('Positive features', ''),
                ('number1', '2.16'),
                ('Negative features', ''),
                ('number0', '-1.14'),
                ('number2', '-0.97'),
                ('<BIAS>', '-0.96'),
                ('hi', '-0.05'),
                ('example', '-0.05'),
            ])
        }

        page_neg, page_pos = request['pages'][:2]
        pred_proba = lambda page: \
            model.predict_proba([extract_text(page['html'])])[0][1]
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
        assert error_response == {
            'id': '3',
            'quality': json.dumps([
                ('Unknown error while training a model',
                 "'bool' object has no attribute 'get'")]),
            'model': None,
        }
    finally:
        producer.send(ATestService.input_topic, {'from-tests': 'stop'})
        producer.flush()
        service_thread.join()


Point = namedtuple('Point', 'x, y')


def test_encode_model():
    p = Point(-1, 2.25)
    assert isinstance(encode_model(p), str)
    assert p == decode_model(encode_model(p))
    assert decode_model(None) is None
    assert encode_model(None) is None


def decode_message(message: bytes) -> Dict:
    try:
        return json.loads(message.decode('utf8'))
    except Exception as e:
        logging.error('Error deserializing message', exc_info=e)
        raise

import logging
import threading

from kafka import KafkaConsumer, KafkaProducer

from hh_page_clf.service import Service, encode_message, decode_message


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s: %(message)s')


class ATestService(Service):
    input_topic = 'test-{}'.format(Service.input_topic)
    ouput_topic = 'test-{}'.format(Service.output_topic)


def clear_topics():
    for topic in [ATestService.input_topic, ATestService.output_topic]:
        consumer = KafkaConsumer(topic)
        while consumer.poll():
            pass


def test_training():
    clear_topics()
    producer = KafkaProducer(value_serializer=encode_message)
    consumer = KafkaConsumer(
        ATestService.output_topic,
        value_deserializer=decode_message)
    service = ATestService()
    service_thread = threading.Thread(target=service.run)
    service_thread.start()
    train_request = {
        'id': 'some id',
        'pages': [
            {
                'url': 'http://example-{}.com/foo'.format(n),
                'html': '<h1>hi example-{} number{}</h1>'.format(n, n % 3),
                'relevant': n % 3 == 1,
            }
            for n in range(10)
        ]
    }
    producer.send(ATestService.input_topic, train_request)
    producer.flush()

    train_response = next(consumer).value

    assert train_response == {
        'id': 'some id',
        'quality': 'Accuracy is 0.84 and some other metric is 0.89',
        'model': 'b64-encoded blob',
    }

    producer.send(ATestService.input_topic, {'from-tests': 'stop'})
    producer.flush()
    service_thread.join()

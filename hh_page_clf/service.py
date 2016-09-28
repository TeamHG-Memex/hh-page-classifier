import argparse
import logging
import json
from typing import Dict

from kafka import KafkaConsumer, KafkaProducer


class Service:
    input_topic = 'dd-modeler-input'
    output_topic = 'dd-modeler-input'

    def __init__(self, kafka_host=None):
        kafka_kwargs = {}
        if kafka_host is not None:
            kafka_kwargs['bootstrap_servers'] = kafka_host
        self.consumer = KafkaConsumer(
            self.input_topic,
            value_deserializer=decode_message,
            **kafka_kwargs)
        self.producer = KafkaProducer(
            value_serializer=encode_message,
            **kafka_kwargs)

    def run(self) -> None:
        for message in self.consumer:
            if message.value == {'from-tests': 'stop'}:
                logging.info('Got message to stop (from tests)')
                break
            self.handle_message(message.value)

    def handle_message(self, message: Dict) -> None:
        print(message)
        logging.info('Got training task with {} pages'.format(
            len(message.get('pages', []))))
        self.send_result({
            'id': message['id'],
            'quality': 'Accuracy is 0.84 and some other metric is 0.89',
            'model': 'b64-encoded blob',
        })

    def send_result(self, result: Dict) -> None:
        logging.info('Sending result for id {}'.format(result.get('id')))
        self.producer.send(self.output_topic, result)
        self.producer.flush()


def encode_message(message: Dict) -> bytes:
    return json.dumps(message).encode('utf8')


def decode_message(message: bytes) -> Dict:
    return json.loads(message.decode('utf8'))


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--kafka-host')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(module)s: %(message)s')
    service = Service(kafka_host=args.kafka_host)
    logging.info('Starting hh page classifier service')
    service.run()

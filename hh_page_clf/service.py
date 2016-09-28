import argparse
import base64
import logging
import json
import pickle
from typing import Dict

from kafka import KafkaConsumer, KafkaProducer

from .train import train_model


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
            self.consumer.commit()

    def handle_message(self, message: Dict) -> None:
        logging.info('Got training task with {} pages'.format(
            len(message.get('pages', []))))
        result = train_model(message['pages'])
        serialized_model = encode_model(result.model)
        logging.info('Sending result for id "{}", model size {} bytes'
                     .format(message.get('id'), len(serialized_model)))
        self.send_result({
            'id': message['id'],
            'quality': result.meta,
            'model': serialized_model,
        })

    def send_result(self, result: Dict) -> None:
        self.producer.send(self.output_topic, result)
        self.producer.flush()


def encode_message(message: Dict) -> bytes:
    try:
        return json.dumps(message).encode('utf8')
    except Exception as e:
        logging.error('Error serializing message', exc_info=e)
        raise


def decode_message(message: bytes) -> Dict:
    try:
        return json.loads(message.decode('utf8'))
    except Exception as e:
        logging.error('Error deserializing message', exc_info=e)
        raise


def encode_model(model: object) -> str:
    return base64.b64encode(pickle.dumps(model, protocol=2)).decode('ascii')


def decode_model(data: str) -> object:
    return pickle.loads(base64.b64decode(data))


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

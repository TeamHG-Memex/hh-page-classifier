import argparse
import base64
import logging
import json
import pickle
from typing import Dict

from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord

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
            consumer_timeout_ms=10,
            **kafka_kwargs)
        self.producer = KafkaProducer(
            value_serializer=encode_message,
            **kafka_kwargs)

    def run(self) -> None:
        to_send = {}
        while True:
            requests = {}  # type: Dict[str, ConsumerRecord]
            order = {}
            for idx, message in enumerate(self.consumer):
                if message.value == {'from-tests': 'stop'}:
                    logging.info('Got message to stop (from tests)')
                    return
                logging.info(
                    'Got training task with {pages} pages, id "{id}", '
                    'message checksum {checksum}, offset {offset}.'
                        .format(
                        pages=len(message.value.get('pages', [])),
                        id=message.value.get('id'),
                        checksum=message.checksum,
                        offset=message.offset,
                    ))
                id_ = message.value['id']
                requests[id_] = message.value
                order[id_] = idx
            self.consumer.commit()
            for id_, result in to_send.items():
                if id_ in requests:
                    logging.info(
                        'Dropping result for id "{}", as new request arrived'
                        .format(id_))
                else:
                    self.send_result(result)
            to_send = {id_: self.train_model(request)
                       for id_, request in sorted(requests.items(),
                                                  key=lambda x: order[x[0]])}

    def train_model(self, request: Dict) -> Dict:
        result = train_model(request['pages'])
        return {
            'id': request['id'],
            'quality': result.meta,
            'model': encode_model(result.model),
        }

    def send_result(self, result: Dict) -> None:
        logging.info('Sending result for id "{}", model size {} bytes'
                     .format(result.get('id'), len(result.get('model', ''))))
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

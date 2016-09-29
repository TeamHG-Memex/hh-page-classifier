#!/usr/bin/env python3
import argparse
import json

from kafka import KafkaProducer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('topic')
    parser.add_argument('filename')
    parser.add_argument('--kafka-host')
    args = parser.parse_args()

    with open(args.filename) as f:
        data = json.dumps(json.load(f)).encode('utf8')

    kafka_kwargs = {}
    if args.kafka_host:
        kafka_kwargs['bootstrap_servers'] = args.kafka_host
    producer = KafkaProducer(**kafka_kwargs)

    producer.send(args.topic, data)
    producer.flush()
    print('Pushed {} bytes to {}'.format(len(data), args.topic))


if __name__ == '__main__':
    main()

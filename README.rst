Headless Horseman Page Classifier
=================================

It gets pages and their labels from The Headless Horseman (THH)
via a kafka queue, trains a model, and sends back both model
and some quality report. The user of THH then might label more pages,
allowing the classifier to reach higher accuracy.

Incoming message example::

    {
      "id": "some id that will be returned in the answer message",
      "pages": [
        {
          "url": "http://example.com",
          "html": "<h1>hi</h1>",
          "relevant": true
        },
        {
          "url": "http://example.com/1",
          "html": "<h1>hi 1</h1>",
          "relevant": false
        },
        {
          "url": "http://example.com/2",
          "html": "<h1>hi 2</h1>",
          "relevant": null
        }
      ]
    }

Outgoing message with trained model::

    {
      "id": "the same id",
      "quality": "Accuracy is 0.84 and some other metric is 0.89",
      "model": "a huge b64-encoded blob?"
    }


Usage
-----

Run the service passing THH host (add hh-kafka to ``/etc/hosts``
if running on a different network)::

    hh-page-clf-service --kafka-host hh-kafka


Testing
-------

Start kafka with zookeper::

    docker run -p 2181:2181 -p 9092:9092 \
        --env ADVERTISED_HOST=127.0.0.1 \
        --env ADVERTISED_PORT=9092 \
        spotify/kafka

Run tests::

    py.test --doctest-modules --cov=hh_page_clf tests hh_page_clf


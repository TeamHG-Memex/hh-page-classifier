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
      "quality": "{ ... }",
      "model": "b64-encoded page classifier model"
    }

``quality`` field is a JSON-encoded string. Here is an example::

    {
     "advice": [
      {
       "kind": "Warning",
       "text": "The quality of the classifier is not very good, ROC AUC is just 0.67. Consider labeling more pages, or re-labeling them using different criteria."
      }
     ],
     "description": [
      {"heading": "Dataset", "text": "183 documents, 183 with labels (100%) across 129 domains."},
      {"heading": "Class balance", "text": "40% relevant, 60% not relevant."},
      {"heading": "Metrics", "text": ""},
      {"heading": "Accuracy", "text": "0.628 ± 0.087"},
      {"heading": "F1", "text": "0.435 ± 0.140"},
      {"heading": "ROC AUC", "text": "0.666 ± 0.127"}
     ],
     "tooltips": {
      "Accuracy": "Accuracy is the ratio of pages classified correctly as relevant or not relevant. This metric is easy to interpret but not very good for unbalanced datasets.",
      "F1": "F1 score is a combination of recall and precision for detecting relevant pages. It shows how good is a classifier at detecting relevant pages at default threshold.Worst value is 0.0 and perfect value is 1.0.",
      "ROC AUC": "Area under ROC (receiver operating characteristic) curve shows how good is the classifier at telling relevant pages from non-relevant at different thresholds. Random classifier has ROC&nbsp;AUC&nbsp;=&nbsp;0.5, and a perfect classifier has ROC&nbsp;AUC&nbsp;=&nbsp;1.0."
     },
     "weights": {
      "neg": [
       {
        "feature": "<BIAS>",
        "hsl_color": "hsl(0, 100.00%, 88.77%)",
        "weight": -1.5918805437501728
       }
      ],
      "neg_remaining": 4006,
      "pos": [
       {
        "feature": "2015",
        "hsl_color": "hsl(120, 100.00%, 80.00%)",
        "weight": 3.630274967418529
       }
      ],
      "pos_remaining": 4513
     }
    }


Usage
-----

Run the service passing THH host (add hh-kafka to ``/etc/hosts``
if running on a different network)::

    hh-page-clf-service --kafka-host hh-kafka


Testing
-------

Install ``pytest`` and ``pytest-cov``.

Start kafka with zookeper::

    docker run --rm -p 2181:2181 -p 9092:9092 \
        --env ADVERTISED_HOST=127.0.0.1 \
        --env ADVERTISED_PORT=9092 \
        spotify/kafka

Run tests::

    py.test --doctest-modules \
        --cov=hh_page_clf --cov-report=term --cov-report=html \
        tests hh_page_clf


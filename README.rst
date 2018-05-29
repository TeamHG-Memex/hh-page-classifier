Headless Horseman Page Classifier
=================================

It gets pages and their labels from Sitehound
(previously The Headless Horseman, or THH)
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


Outgoing message with progress (``dd-modeler-progress`` queue)::

  {
    "id": "some id",
    "percentage_done": 98.123,
  }


Usage
-----

Run the service passing THH host (add hh-kafka to ``/etc/hosts``
if running on a different network)::

    hh-page-clf-service --kafka-host hh-kafka


LDA model trained on 500k dmoz pages with bigrams and 100k features,
and random pages (1k alexa top-1m sample) are available at
``s3://darpa-memex/thh/``: ``random-pages.jl.gz`` and ``lda.pkl``.

Pass path to random pages via ``--random-pages`` argument, and path to LDA
model via ``--lda`` argument. Note that LDA model is optional and is disabled
by default. It can get a very slight improvement in accuracy and some sensible
looking topics, but also slows down training and prediction quite a bit,
and requires more memory.

Random pages are a sample of a low number (about 1k) random pages
in the same format as input pages (with "url" and "html" fields).
They are used as negative examples during training.

An LDA model was trained on a large number of random pages. It's features are
used in addition to text features from the page. You may build an LDA model
yourself (see also command line options, good results can be obtained
with 300 topics with bigrams)::

    train-lda text-items.jl.gz lda.joblib

For faster loading, it is recommended to re-dump the model with pickle
(joblib can load pickled data as well).


Building docker image
---------------------

Building does not require anything special, just check out the project and run::

    docker build -t hh-page-clf .


Accuracy testing
----------------

If you have some datasets in json format (they may be gzipped), you can check
accuracy, eli5 work and serialization by running::

    hh-page-clf-train my-dataset.json.gz --lda lda.pkl

or even run on several datasets and see an aggregate accuracy report::

    hh-page-clf-train datasets/*.json.gz --lda lda.pkl


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
        --ignore=hh_page_clf/pretraining \
        tests hh_page_clf

Cleaning Kafka queues at the start of tests/test_service.py can
*sometimes* hang - just try once again.

----

.. image:: https://hyperiongray.s3.amazonaws.com/define-hg.svg
	:target: https://www.hyperiongray.com/?pk_campaign=github&pk_kwd=hh-page-classifier
	:alt: define hyperiongray

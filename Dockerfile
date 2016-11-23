FROM python:3.5

WORKDIR /opt/hh-page-clf

RUN apt-get update && apt-get install -y tree

COPY requirements.txt .
RUN pip install -U pip wheel && \
    pip install -r requirements.txt

COPY random-pages.jl.gz .

COPY setup.py .
COPY hh_page_clf hh_page_clf
RUN tree
RUN pip install -e .

CMD hh-page-clf-service --kafka-host hh-kafka
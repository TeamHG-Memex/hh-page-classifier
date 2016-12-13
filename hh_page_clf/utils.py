import base64
import logging
from itertools import chain
import pickle
from typing import Optional
import zlib

import stop_words


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(module)s: %(message)s')


def encode_object(model: object) -> str:
    if model is not None:
        return (
            base64.b64encode(
                zlib.compress(
                    pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)))
            .decode('ascii'))


def decode_object(data: Optional[str]) -> object:
    if data is not None:
        return pickle.loads(zlib.decompress(base64.b64decode(data)))


_stop_words = None


def get_stop_words():
    global _stop_words
    if _stop_words is None:
        _stop_words = set(chain.from_iterable(
            stop_words.get_stop_words(lang)
            for lang in stop_words.AVAILABLE_LANGUAGES
        ))
    return _stop_words

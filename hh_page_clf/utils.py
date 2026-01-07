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


class PBar:
    def __init__(self, callback=None,
                 parent: 'PBar'=None, parent_ratio: float=None):
        self._done_units = 0.0
        self._callback = callback
        self._parent = parent
        self._parent_ratio = parent_ratio
        if parent is not None and not parent_ratio:
            raise ValueError('set non-zero parent_ratio')

    def progress(self, units):
        self._done_units += units
        if self._parent is not None:
            self._parent.progress(units * self._parent_ratio)
        if self._callback is not None:
            self._callback(self._done_units)

    def make_child(self, parent_ratio: float) -> 'PBar':
        return PBar(parent=self, parent_ratio=parent_ratio)

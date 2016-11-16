import base64
import logging
import pickle
from typing import Optional
import zlib


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(module)s: %(message)s')


def encode_object(model: object) -> str:
    return (
        base64.b64encode(
            zlib.compress(
                pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)))
        .decode('ascii'))


def decode_object(data: Optional[str]) -> object:
    if data is not None:
        return pickle.loads(zlib.decompress(base64.b64decode(data)))

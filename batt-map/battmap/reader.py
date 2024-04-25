from pathlib import Path

from .utils import get_test_id, get_mode


def get_key(file):
    file = Path(file)
    mode = get_mode(file)
    test_id = get_test_id(file)
    return f'{test_id}_{mode}'


class ReaderCollection:
    def __init__(self):
        self.readers = {}

    def add_reader(self, key, reader):
        self.readers[key] = reader

    def __call__(self, file):
        key = get_key(file)
        reader = self.readers.get(key, None)
        if reader is None:
            raise KeyError(f'No reader defined for key {key}')

        return reader(file)

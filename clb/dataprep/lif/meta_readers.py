"""Module defines methods for reading metadata and corresponding data.

Methods are stored in class MetaReader, their behavior can be customized by
implementing different interfaces.
"""
from abc import ABC, abstractmethod
from functools import partial
from itertools import tee


class MetaReader(ABC):
    """Class defines methods for reading metadata and corresponding data."""
    def __init__(self, path):
        self.path = path

    @abstractmethod
    def close(self):
        """Close connection with file."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @abstractmethod
    def meta_reader(self):
        """Read metadata.

        Returns:
            iterable: Metadata of file loaded from `path` parameter.
        """

    @staticmethod
    @abstractmethod
    def check_meta_params_values(meta, params):
        """Tell if given parameters in `meta` have given values.

        Args:
            meta: Metadata.
            params (dict): Parameters to check.

        Returns:
            bool: True if parameters in `meta` have values like in `params`
                  and False otherwise.
        """

    def data_reader(self, meta):
        """Read data items.

        Assumes information in meta are suitable to locate corresponding data.

        Args:
            meta: Metadata.

        Returns:
            Data item to which `meta` belongs.
        """

    def get_matching_meta(self, params):
        """Get metadata of items, that match given parameters.

        `meta_reader` is used for reading metadata of items.
        `check_meta_params_values` is used to check for match.

        Args:
            params (dict): Parameters to check for match.

        Returns:
            filter generator object: Metadata of items that match given
                                     parameters.
        """
        # Reading all metadata.
        meta = self.meta_reader()

        # Filter out not matching.
        is_matching = partial(self.check_meta_params_values, params=params)
        matching_meta = filter(is_matching, meta)

        return matching_meta

    def read_data_given_meta(self, params, with_meta=True):
        """Read data items, which metadata match `params`.

        First metadata that match given parameters is read with
        `get_matching_meta`. Then for each of those metadata pieces
        corresponding data item is read with `data_reader`.

        Args:
            params (dict): Parameters that will be checked for match.
            with_meta (bool): Should metadata be also returned.

        Returns:
            zip generator object: Metadata, data.
        """
        # Reading metadata of matching items.
        matching_meta, meta_copy = tee(self.get_matching_meta(params))

        # Reading items corresponding to metadata.
        data = map(self.data_reader, matching_meta)

        # Adding metadata to output if required.
        if with_meta:
            out = zip(meta_copy, data)
        else:
            out = data

        return out


if __name__ == '__main__':
    # Examples.

    # This is our data, first element of each tuple is metadata, second
    # actual data.
    data = ((dict(index=0, sample=1, speed=50, channels=2), (1, 2)),
            (dict(index=1, sample=2, speed=50, channels=3), (1, 2, 3))
            )

    class DummyReader(MetaReader):
        def open(self):
            pass

        def close(self):
            pass

        def meta_reader(self):
            for d in data:
                yield d[0]

        @staticmethod
        def check_meta_params_values(meta, params):
            meta_checkers = {
                'index': lambda x, y: x['index'] == y,
                'sample': lambda x, y: x['sample'] == y,
                'speed': lambda x, y: x['speed'] == y,
            }

            return all(meta_checkers[name](meta, value)
                       for name, value in params.items())

        def data_reader(self, meta):
            return data[meta['index']][1]

    reader = DummyReader('dummy_path')

    # Let's say we want to check if there is image with speed 50 and sample 1.
    print(tuple(reader.get_matching_meta(dict(speed=50, sample=1))))

    # If we want to read this image, we can just call data_reader with our
    # matching meta. But we can also do it with one call.
    print(tuple(reader.read_data_given_meta(dict(speed=50, sample=1))))

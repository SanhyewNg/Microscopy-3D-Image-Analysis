import pytest
import unittest

from clb.dataprep.lif.meta_readers import MetaReader


class Reader(MetaReader):
    def __init__(self, data):
        super().__init__(data)
        self.data = data

    def open(self):
        super().open()

    def close(self):
        super().close()

    def meta_reader(self):
        for d in self.data:
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
        return self.data[meta['index']][1]


@pytest.mark.io
class TestMetaReader(unittest.TestCase):
    def setUp(self):
        data = ((dict(index=0, sample=1, speed=50, channels=2), (1, 2)),
                (dict(index=1, sample=2, speed=50, channels=3), (1, 2, 3))
                )
        self.reader = Reader(data)

    def test_get_matching_meta_one_match(self):
        output = tuple(self.reader.get_matching_meta(dict(index=0)))
        right_output = (dict(index=0, sample=1, speed=50, channels=2), )

        self.assertTupleEqual(output, right_output)

    def test_get_matching_meta_two_matches(self):
        output = tuple(self.reader.get_matching_meta(dict(speed=50)))
        right_output = (dict(index=0, sample=1, speed=50, channels=2),
                        dict(index=1, sample=2, speed=50, channels=3))

        self.assertTupleEqual(output, right_output)

    def test_read_data_given_meta_one_match(self):
        output = tuple(self.reader.read_data_given_meta(dict(sample=1)))

        right_output = (
            (dict(index=0, sample=1, speed=50, channels=2), (1, 2)),
        )

        self.assertTupleEqual(output, right_output)

    def test_read_data_given_meta_two_matches(self):
        output = tuple(self.reader.read_data_given_meta(dict(speed=50)))
        right_output = (
            (dict(index=0, sample=1, speed=50, channels=2), (1, 2)),
            (dict(index=1, sample=2, speed=50, channels=3), (1, 2, 3))
        )

        self.assertTupleEqual(output, right_output)


if __name__ == '__main__':
    unittest.main()

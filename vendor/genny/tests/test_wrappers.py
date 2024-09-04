import unittest
from vendor.genny.genny.wrappers import GeneratorWrapper, gen_wrapper
from itertools import islice, tee


class TestWrappedGenerator(unittest.TestCase):
    def test_if_raises_stop_iteration(self):
        gen = GeneratorWrapper(range, 0)
        with self.assertRaises(StopIteration):
            next(gen)

    def test_if_raises_stop_iteration_on_second_time(self):
        gen = GeneratorWrapper(range, 5)
        for _ in gen:
            pass

        with self.assertRaises(StopIteration):
            next(gen)

    def test_with_one_range(self):
        gen = GeneratorWrapper(range, 5)
        real_gen = range(5)

        self.assertTupleEqual(tuple(gen), tuple(real_gen))

    def test_with_range_and_islice(self):
        gen = range(5) | GeneratorWrapper(islice, 2, 3)
        real_gen = (2, )

        self.assertTupleEqual(tuple(gen), tuple(real_gen))

    def test_with_two_generators(self):
        @gen_wrapper
        def gen1(n):
            for i in range(n):
                yield i

        @gen_wrapper
        def gen2(gen, k):
            for i in gen:
                yield i + k

        gen = gen1(5) | gen2(3)
        real_gen = (3, 4, 5, 6, 7)

        self.assertTupleEqual(tuple(gen), tuple(real_gen))

    def test_with_tee(self):
        # Now it should be (1, 2)
        gen = range(5) | GeneratorWrapper(islice, 1, 3)

        gen1, gen2 = tee(gen)

        @gen_wrapper
        def generator(gen, k):
            for i in gen:
                yield i + k

        gen1 |= generator(3)
        gen2 |= generator(5)

        real_gen1 = (4, 5)
        real_gen2 = (6, 7)

        self.assertTupleEqual(tuple(gen1), real_gen1)
        self.assertTupleEqual(tuple(gen2), real_gen2)

    def test_with_iterable_on_the_left_side(self):
        left = [1, 2, 3, 4]

        @gen_wrapper
        def generator(gen, n):
            for i in gen:
                yield i + n

        piped = left | generator(3)
        real_piped = (4, 5, 6, 7)

        self.assertTupleEqual(tuple(piped), real_piped)


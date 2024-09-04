from functools import partial, wraps


class GeneratorWrapper:
    """Wrapper around generators to pipe them conveniently using ' | ' symbol.

    It requires all piped generators to have incoming generator as the first
    argument (if generator is a separate function) or the second one (if
    generator is part of a class and has `self` as the first argument).

    First generator in a pipe is assumed not to rely on any other generator.
    If one wants to use some external generator for it, it has to be passed
    manually as it's argument, e.g.:
    ```
    @gen_wrapper
    def first_gen(external_gen, ...):
        ...
    ```
    Every other generator in a pipe (second, third, etc.) is assumed to take
    generator as a first argument, e.g.:
    ```
    @gen_wrapper
    def second_generator(data_gen, ...):
        ...
    ```
    This argument is automatically filled by GeneratorWrapper class when
    piping generator calls with ' | ' symbol.

    Attributes:
        cached_generator_func (func): cached generator function. It is saved
                                      in GeneratorWrapper object to delay
                                      its call. It will be called when
                                      iterating over GeneratorWrapper
                                      object.
        args (tuple): Arguments to generator function.
        kwargs (dict): Keyword arguments to generator function.
        wrapped_generator (iterator): Generator object returned by
                                      `cached_generator_func`. It is created
                                      in order to ensure that after being
                                      exhausted (all data has been yielded),
                                      it's not possible to iterate over
                                      GeneratorWrapper object anymore
                                      (StopIteration is raised).
    """
    def __init__(self, generator_function_to_cache, *args, **kwargs):
        self.cached_generator_func = generator_function_to_cache
        self.args = args
        self.kwargs = kwargs
        self.wrapped_generator = None

    def __iter__(self):
        """Make it possible to iterate over GeneratorWrapper object.
        """
        return self

    def __next__(self):
        """Make it possible to run next() on GeneratorWrapper object.
        """
        if self.wrapped_generator is None:
            # When iterating over GeneratorWrapper object for the first time,
            # self.wrapped_generator object doesn't exist. Create it here.
            self.wrapped_generator = iter(self.cached_generator_func(*self.args,
                                                                     **self.kwargs))

        return next(self.wrapped_generator)

    def __ror__(self, left_generator):
        """Overload '|' symbol.

        It is used when left object to the '|' symbol is of different type
        than the right object to the '|' symbol.

        Args:
            left_generator: generator to the left_generator of '|' symbol.
                            Can be an iterable of any type (not exactly
                            GeneratorWrapper object, can be e.g. a list).

        Returns:
            an object of type GeneratorWrapper that is piped with
            `left_generator`.
        """
        # Create an object: left_generator is the generator that will be fed
        # into the next generator as the first argument.
        return self.__class__(self.cached_generator_func, left_generator,
                              *self.args, **self.kwargs)

    def __or__(self, right_generator):
        """Overload '|' symbol.

        It is used when both objects (to the left and to the right) used
        with '|' symbol are of GeneratorWrapper type.

        Args:
            right_generator: generator to the right of '|' symbol. It has to
                             be of type GeneratorWrapper.
        """
        return right_generator.__ror__(self)


def gen_wrapper(f):
    """Wrapper for a generator being a separate function.

    It makes it possible to use generator() instead of
    GeneratorWrapper(generator). It is essential to wrap EACH chained
    generator with `@gen_wrapper` command, e.g.:
    ```
    @gen_wrapper
    def some_generator(...):
        ...
    ```

    If no wrapping is possible (e.g.
    when using external generator), then wrap it like this:
    ```
    gen_chain = (gen1(...) |
                 gen2(...) |
                 GeneratorWrapper(external_gen,
                                  [external gen args separated by comma]) |
                                   ...)
    ```
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        return GeneratorWrapper(f, *args, **kwargs)

    return wrapper


def obj_gen_wrapper(f):
    """Wrapper for a generator belonging to a separate class (as a method).

    It makes it possible to use generator() instead of
    GeneratorWrapper(generator). It is essential to wrap EACH chained
    generator that is part of external object with `@obj_gen_wrapper`
    command, e.g.:
    ```
    class Augmentator():
        ...
        @obj_gen_wrapper
        def flow(self, data_gen, ...):
            ...
            yield ...

    augmentator = Augmentator()
    gen_chain = (gen1(...) |
                 gen2(...) |
                 augmentator.flow(...) |
                 ...)
    ```
    It assumes that right after `self` there's
    an argument that takes some generator as input to the current generator.
    """
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        #  Extract self parameter to provide generator as 'first' one.
        g = partial(f, self)
        return GeneratorWrapper(g, *args, **kwargs)

    return wrapper

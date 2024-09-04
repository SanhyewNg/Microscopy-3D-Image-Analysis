import numpy as np


class LayersFiltersIterator:

    def __init__(self, num_layers, start_filter_num, going_down):
        """An iterator called by __iter__ method of LayersNum class object,
        that iterators over number of filters for a number of specified
        levels in a arm of an U-net. The direction is specified by the
        going_down flag.

        Args:
            num_layers (int): Number of layers in the arm of a U-net.
            start_filter_num (int): Number of filters at the shallowest
            level of the U-net.
            going_down (bool): A flag stating whether iterator should go
            from shallower layers into the deeper layers, or in the other
            direction.
        """
        self.filter_exponent = int(np.log2(start_filter_num)) - 1
        self.going_down = going_down
        if going_down:
            self.current_layer = 1
            self.num_layers = num_layers
        else:
            self.current_layer = num_layers
            self.num_layers = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.going_down:
            if self.current_layer <= self.num_layers:
                current_layer = self.current_layer
                current_feature = 2 ** (self.filter_exponent +
                                        current_layer)
                self.current_layer += 1
                return current_layer, current_feature
            else:
                raise StopIteration()
        else:
            if self.current_layer >= self.num_layers:
                current_layer = self.current_layer
                current_feature = 2 ** (self.filter_exponent +
                                        current_layer)
                self.current_layer -= 1
                return current_layer, current_feature
            else:
                raise StopIteration()

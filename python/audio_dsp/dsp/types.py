# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np

import audio_dsp.dsp.utils as utils


class float32:
    def __init__(self, value):
        self.value = np.float32(value)
        return

    def __call__(self):
        return self.value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        yield self.value

    def __abs__(self):
        return float32(np.abs(self.value))

    def __gt__(self, other_f32):
        if isinstance(other_f32, float32):
            return self.value > other_f32.value
        else:
            raise TypeError("float32 can only be compared against float32")

    def __lt__(self, other_f32):
        if isinstance(other_f32, float32):
            return self.value < other_f32.value
        else:
            raise TypeError("float32 can only be compared against float32")

    def __ge__(self, other_f32):
        if isinstance(other_f32, float32):
            return self.value >= other_f32.value
        else:
            raise TypeError("float32 can only be compared against float32")

    def __sub__(self, other_f32):
        if isinstance(other_f32, float32):
            return float32(self.value - other_f32.value)
        else:
            raise TypeError("float32 can only be subtracted from float32")

    def __add__(self, other_f32):
        if isinstance(other_f32, float32):
            return float32(self.value + other_f32.value)
        else:
            raise TypeError("float32 can only be added to float32")

    def __mul__(self, other_f32):
        if isinstance(other_f32, float32):
            return float32(self.value * other_f32.value)
        else:
            raise TypeError("float32 can only be multiplied with float32")

    def __truediv__(self, other_f32):
        if isinstance(other_f32, float32):
            return float32(self.value / other_f32.value)
        else:
            raise TypeError("float32 can only be multiplied with float32")

    def __float__(self):
        return float(self.value)

    def __pow__(self, other_f32):
        if isinstance(other_f32, float32):
            return float32(self.value**other_f32.value)
        else:
            raise TypeError("float32 can only be raised to the power of float32")

    def as_int32(self):
        return utils.int32(self.value)

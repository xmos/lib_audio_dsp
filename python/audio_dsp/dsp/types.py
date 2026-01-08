# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Custom Python types used by DSP blocks."""

import numpy as np
import math

import audio_dsp.dsp.utils as utils


class float32:
    """This class utilises numpy.float32, but restricts numerical
    operations to other float32 types to avoid promotion to other
    floating point types.
    """

    def __init__(self, value):
        self.value = np.float32(value)
        return

    def __call__(self):
        """Return the float32 value."""
        return self.value

    def __str__(self):
        """Return the float32 value as a string."""
        return str(self.value)

    def __repr__(self):
        """Return the float32 value as a string."""
        return self.__str__()

    def __iter__(self):
        """Return the float32 value during an interable call."""
        yield self.value

    def __abs__(self):
        """Return the magnitude of the float32 value as a float32."""
        return float32(np.abs(self.value))

    def __neg__(self):
        """Return the negative of the float32 value as a float32."""
        return float32(-self.value)

    def __gt__(self, other_f32):
        """Calculate if the float32 is greater than the other float32."""
        if isinstance(other_f32, float32):
            return self.value > other_f32.value
        if isinstance(other_f32, float):
            return self.value > float32(other_f32).value
        else:
            raise TypeError("float32 can only be compared against float32 or float")

    def __lt__(self, other_f32):
        """Calculate if the float32 is less than the other float32."""
        if isinstance(other_f32, float32):
            return self.value < other_f32.value
        if isinstance(other_f32, float):
            return self.value < float32(other_f32).value
        else:
            raise TypeError("float32 can only be compared against float32 or float")

    def __ge__(self, other_f32):
        """Calculate if the float32 is greater than or equal to the
        other float32 value.
        """
        if isinstance(other_f32, float32):
            return self.value >= other_f32.value
        elif isinstance(other_f32, float):
            return self.value >= float32(other_f32).value
        else:
            raise TypeError("float32 can only be compared against float32 or float")

    def __sub__(self, other_f32):
        """Subtract one float32 value from another."""
        if isinstance(other_f32, float32):
            return float32(self.value - other_f32.value)
        else:
            raise TypeError("float32 can only be subtracted from float32")

    def __add__(self, other_f32):
        """Add one float32 value to another."""
        if isinstance(other_f32, float32):
            return float32(self.value + other_f32.value)
        else:
            raise TypeError("float32 can only be added to float32")

    def __mul__(self, other_f32):
        """Multiply one float32 value by another."""
        if isinstance(other_f32, float32):
            return float32(self.value * other_f32.value)
        else:
            raise TypeError("float32 can only be multiplied with float32")

    def __truediv__(self, other_f32):
        """Divide one float32 value by another."""
        if isinstance(other_f32, float32):
            return float32(self.value / other_f32.value)
        else:
            raise TypeError("float32 can only be multiplied with float32")

    def __float__(self):
        """Return the float32 as a native Python float."""
        return float(self.value)

    def __pow__(self, other_f32):
        """Calculate x^y for a float32."""
        if isinstance(other_f32, float32):
            return float32(self.value**other_f32.value)
        else:
            raise TypeError("float32 can only be raised to the power of float32")

    def as_int32(self):
        """Convert the float32 value to int32 format."""
        return utils.int32(float(self.value))


class float_s32:
    """A floating point number type, where the maintissa is a 32 bit
    number and the exponent is an integer.
    """

    def __init__(self, value, Q_sig=None):
        self.Q_sig = Q_sig
        if Q_sig and isinstance(value, float):
            self.mant = utils.int32(round(value * (1 << int(Q_sig))))
            self.exp = -Q_sig
        elif isinstance(value, (float, np.float32)):
            self.mant, self.exp = math.frexp(value)
            self.mant = utils.float_to_int32(self.mant)
            self.exp -= 31
        elif isinstance(value, int):
            self.mant, self.exp = math.frexp(float(value))
            self.mant = utils.float_to_int32(self.mant)
            self.exp -= 31
        elif isinstance(value, list):
            self.mant = utils.int32(value[0])
            self.exp = utils.int32(value[1])
        else:
            TypeError("s32 can only be initialised by float or list of ints [mant, exp]")

        # overflow checks
        self.mant = utils.int32(self.mant)
        self.exp = utils.int32(self.exp)

    def __mul__(self, other_s32):
        """Multiply one float_s32 by one another."""
        if isinstance(other_s32, float_s32):
            # vect_s32_mul_prepare
            b_hr = utils.hr_s32(self)
            c_hr = utils.hr_s32(other_s32)
            total_hr = b_hr + c_hr

            if total_hr == 0:
                b_shr = 1
                c_shr = 1
            elif total_hr == 1:
                b_shr = 1 if b_hr == 0 else 0
                c_shr = 1 if c_hr == 0 else 0
            elif b_hr == 0:
                b_shr = 0
                c_shr = 2 - total_hr
            elif c_hr == 0:
                b_shr = 2 - total_hr
                c_shr = 0
            else:
                b_shr = 1 - b_hr
                c_shr = 1 - c_hr

            res_exp = self.exp + other_s32.exp + b_shr + c_shr + 30

            # vect_s32_mul
            B = utils.ashr32(self.mant, b_shr)
            C = utils.ashr32(other_s32.mant, c_shr)

            A = utils.vpu_mult(B, C)

            return float_s32([A, res_exp])
        else:
            raise TypeError("s32 can only be multiplied by s32")

    def __truediv__(self, other_s32):
        """Divide one float_s32 by one another."""
        if isinstance(other_s32, float_s32):
            # float_s32_div and s32_inverse
            b_hr = utils.hr_s32(other_s32)
            scale = 2 * 30 - b_hr
            dividend = 1 << scale
            t = float_s32([dividend / other_s32.mant, -scale - other_s32.exp])

            return self.__mul__(t)

        else:
            raise TypeError("s32 can only be divided by s32")

    def __add__(self, other_s32):
        """Add float_s32s to one another."""
        if isinstance(other_s32, float_s32):
            # from float_s32_add in lib_xcore_math
            x_hr = utils.hr_s32(self)
            y_hr = utils.hr_s32(other_s32)
            x_min_exp = self.exp - x_hr
            y_min_exp = other_s32.exp - y_hr

            res_exp = max(x_min_exp, y_min_exp) + 1

            x_shr = res_exp - self.exp
            y_shr = res_exp - other_s32.exp

            mant = utils.ashr32(self.mant, x_shr) + utils.ashr32(other_s32.mant, y_shr)

            return float_s32([mant, res_exp])
        else:
            raise TypeError("s32 can only be added to s32")

    def __sub__(self, other_s32):
        """Subtract float_s32s from one another."""
        if isinstance(other_s32, float_s32):
            # from float_s32_sub in lib_xcore_math
            x_hr = utils.hr_s32(self)
            y_hr = utils.hr_s32(other_s32)
            x_min_exp = self.exp - x_hr
            y_min_exp = other_s32.exp - y_hr

            res_exp = max(x_min_exp, y_min_exp) + 1

            x_shr = res_exp - self.exp
            y_shr = res_exp - other_s32.exp

            mant = utils.ashr32(self.mant, x_shr) - utils.ashr32(other_s32.mant, y_shr)

            return float_s32([mant, res_exp])
        else:
            raise TypeError("s32 can only be subtracted from s32")

    def __gt__(self, other_s32):
        """Calculate greater than between 2 float_s32s."""
        if isinstance(other_s32, float_s32):
            delta = self.__sub__(other_s32)
            return delta.mant > 0
        else:
            raise TypeError("s32 can only be compared against s32")

    def __lt__(self, other_s32):
        """Calculate less than between 2 float_32s."""
        if isinstance(other_s32, float_s32):
            delta = self.__sub__(other_s32)
            return delta.mant < 0
        else:
            raise TypeError("s32 can only be compared against s32")

    def __abs__(self):
        """Calculate absolute value of a float_s32."""
        return float_s32([abs(self.mant), self.exp])

    def __float__(self):
        """Cast a float_s32 to native Python float64.
        Add 31 here; Python expects a float mantissa < 1, but we use utils.int32.
        """
        assert isinstance(self.mant, int)
        return math.ldexp(utils.int32_to_float(self.mant), self.exp + 31)

    __rmul__ = __mul__

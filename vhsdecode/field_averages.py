from vhsdecode.utils import StackableMA
from collections import deque


class FieldAverage:
    def __init__(self, chroma_agc_fields, chroma_transfer_fields=30):
        self._rf_level = StackableMA()

        self.chroma_level_len = chroma_agc_fields
        self._chroma_level_even = deque(maxlen=self.chroma_level_len)
        self._chroma_level_odd = deque(maxlen=self.chroma_level_len)

        # Measured luma-to-color-under amplitude transfer, as (band centres,
        # response) per field. The regression behind it is noisy in any single
        # field, but the transfer is a property of the head and the tape - and
        # of WHICH head, so the two are kept apart the same way the chroma gain
        # above is. Measured, they differ by 15% of the transfer's own level,
        # resolved at 3.4 sigma in the mid bands.
        self._chroma_transfer_even = deque(maxlen=chroma_transfer_fields)
        self._chroma_transfer_odd = deque(maxlen=chroma_transfer_fields)

        # self.line_length = StackableMA()
        # self.vsync_dist = StackableMA

    @property
    def rf_level(self):
        return self._rf_level

    # even field state for the chroma automatic gain control
    @property
    def chroma_level_even(self):
        return self._chroma_level_even
    
    # odd field state for the chroma automatic gain control
    @property
    def chroma_level_odd(self):
        return self._chroma_level_odd

    # measured luma-to-color-under amplitude transfer, per head
    def chroma_transfer_for(self, is_first_field):
        return self._chroma_transfer_odd if is_first_field else self._chroma_transfer_even


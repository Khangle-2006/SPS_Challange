"""Versioned dataset adapters: PPG convention is established at the backend boundary.

The source snapshot and its access gates are preserved. Global MCD harmonization
contains no camera/split condition, so future authorized dataset partitions use the
same normalization. This module does not open additional partitions.
"""
import source_data_adapters as source
from ppg_polarity import normalize_recording

Label = source.Label
Recording = source.Recording
VideoFrames = source.VideoFrames
ImageFrames = source.ImageFrames
V3Frames = source.V3Frames
ExcludedRecording = source.ExcludedRecording
check_time = source.check_time
video_pts = source.video_pts
read_timed_values = source.read_timed_values


class _PolarityBoundary:
    def load(self, *args, **kwargs):
        return normalize_recording(super().load(*args, **kwargs))


class UBFC2(_PolarityBoundary, source.UBFC2):
    pass


class MCD(_PolarityBoundary, source.MCD):
    pass


class UBFC1(_PolarityBoundary, source.UBFC1):
    pass


class PURE(_PolarityBoundary, source.PURE):
    pass


class AiPEX(_PolarityBoundary, source.AiPEX):
    pass


class Pilot(_PolarityBoundary, source.Pilot):
    pass


BACKENDS = {'ubfc2': UBFC2, 'mcd': MCD, 'ubfc1': UBFC1,
            'pure': PURE, 'aipex': AiPEX, 'pilot': Pilot}

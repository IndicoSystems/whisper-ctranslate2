import unittest

import numpy as np

from whisper_ctranslate2.diarization import Diarization


class TestDiarization(unittest.TestCase):
    class Segment:
        start: float = 0.0
        end: float = 0.0

        def __init__(self, start, end):
            self.start = start
            self.end = end

    def _convert_pyannote_data_to_records(self, pyannote_data):
        """Convert pyannote data format to numpy structured array format"""
        date_frame = np.array(
            pyannote_data,
            dtype=[("segment", object), ("label", object), ("speaker", object)],
        )

        segments_as_records = np.core.records.fromarrays(
            [
                date_frame["segment"],
                date_frame["label"],
                date_frame["speaker"],
                np.array([seg.start for seg in date_frame["segment"]]),
                np.array([seg.end for seg in date_frame["segment"]]),
                np.zeros(len(date_frame)),
            ],
            names="segment, label, speaker, start, end, intersection",
        )

        return segments_as_records

    def test_no_speaker(self):
        pyannote_data = [(TestDiarization.Segment(10, 20), "A", "SPEAKER_00")]
        diarize_data = self._convert_pyannote_data_to_records(pyannote_data)

        segment = {"start": 1, "end": 5}
        segments = [segment]

        result = Diarization()._do_assign_speakers_to_segments(
            diarize_data, {"segments": segments}, None
        )

        self.assertFalse("speaker" in result["segments"][0])

    def test_single_speaker(self):
        pyannote_data = [(TestDiarization.Segment(2, 4), "A", "SPEAKER_00")]
        diarize_data = self._convert_pyannote_data_to_records(pyannote_data)

        segment = {"start": 1, "end": 5}
        segments = [segment]

        result = Diarization()._do_assign_speakers_to_segments(
            diarize_data, {"segments": segments}, None
        )

        self.assertEqual("SPEAKER_00", result["segments"][0]["speaker"])

    def test_two_speakers(self):
        pyannote_data = [
            (TestDiarization.Segment(1, 5), "A", "SPEAKER_00"),
            (TestDiarization.Segment(5, 7), "B", "SPEAKER_01"),
        ]
        diarize_data = self._convert_pyannote_data_to_records(pyannote_data)

        segment = {"start": 4, "end": 10}
        segments = [segment]

        result = Diarization()._do_assign_speakers_to_segments(
            diarize_data, {"segments": segments}, None
        )

        self.assertEqual("SPEAKER_01", result["segments"][0]["speaker"])

    def test_single_speaker_with_speakername(self):
        SPEAKER_NAME = "PARLANT"
        pyannote_data = [(TestDiarization.Segment(2, 4), "A", "SPEAKER_00")]
        diarize_data = self._convert_pyannote_data_to_records(pyannote_data)

        segment = {"start": 1, "end": 5}
        segments = [segment]

        result = Diarization()._do_assign_speakers_to_segments(
            diarize_data, {"segments": segments}, SPEAKER_NAME
        )

        self.assertEqual("PARLANT_00", result["segments"][0]["speaker"])


if __name__ == "__main__":
    unittest.main()

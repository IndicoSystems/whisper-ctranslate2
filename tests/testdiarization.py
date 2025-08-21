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

    def create_diarize_df(self, segments_data):
        """Helper method to create the correct diarize_df format"""
        diarize_data = []
        for start, end, label, speaker in segments_data:
            segment = self.Segment(start, end)
            diarize_data.append((segment, label, speaker))
        
        date_frame = np.array(
            diarize_data,
            dtype=[
                ("segment", object),
                ("label", object),
                ("speaker", object)
            ],
        )

        segments_as_records = np.rec.fromarrays(
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
        # No diarization data overlapping with segment
        diarize_df = self.create_diarize_df([
            (10, 20, "A", "SPEAKER_00")
        ])

        segment = {"start": 1, "end": 5}
        segments = [segment]

        result = Diarization()._do_assign_speakers_to_segments(
            diarize_df, {"segments": segments}, None
        )

        print(result)

        self.assertFalse("speaker" in result["segments"][0])

    def test_single_speaker(self):
        diarize_df = self.create_diarize_df([
            (2, 4, "A", "SPEAKER_00")
        ])

        segment = {"start": 1, "end": 5}
        segments = [segment]


        result = Diarization()._do_assign_speakers_to_segments(
            diarize_df, {"segments": segments}, None
        )

        print(result)

        self.assertEqual("SPEAKER_00", result["segments"][0]["speaker"])

    def test_two_speakers(self):
        diarize_df = self.create_diarize_df([
            (1, 5, "A", "SPEAKER_00"),
            (5, 7, "B", "SPEAKER_01"),
        ])

        segment = {"start": 4, "end": 10}
        segments = [segment]

        result = Diarization()._do_assign_speakers_to_segments(
            diarize_df, {"segments": segments}, None
        )

        print(result)

        # The segment overlaps more with SPEAKER_01 (5-7) than SPEAKER_00 (4-5)
        self.assertEqual("SPEAKER_01", result["segments"][0]["speaker"])

    def test_single_speaker_with_speakername(self):
        SPEAKER_NAME = "PARLANT"
        diarize_df = self.create_diarize_df([
            (2, 4, "A", "SPEAKER_00")
        ])

        segment = {"start": 1, "end": 5}
        segments = [segment]

        result = Diarization()._do_assign_speakers_to_segments(
            diarize_df, {"segments": segments}, SPEAKER_NAME
        )

        print(result)

        self.assertEqual("PARLANT_00", result["segments"][0]["speaker"])


if __name__ == "__main__":
    unittest.main()

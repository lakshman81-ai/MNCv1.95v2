import unittest
from pathlib import Path

from backend.transcription import transcribe_audio_pipeline


class TranscriptionMockTests(unittest.TestCase):
    def setUp(self):
        self.mock_xml_path = Path(__file__).resolve().parents[1] / "mock_data" / "happy_birthday.xml"
        self.placeholder_audio = str(self.mock_xml_path)

    def test_mock_musicxml_matches_fixture(self):
        result = transcribe_audio_pipeline(self.placeholder_audio, use_mock=True)
        expected_xml = self.mock_xml_path.read_text(encoding="utf-8")

        self.assertEqual(result.musicxml.strip(), expected_xml.strip())
        self.assertEqual(result.analysis_data.meta.sample_rate, 22050)

    def test_mock_pipeline_returns_analysis_object(self):
        result = transcribe_audio_pipeline(self.placeholder_audio, use_mock=True)

        self.assertIsNotNone(result.analysis_data)
        self.assertEqual(result.analysis_data.timeline, [])
        self.assertEqual(result.analysis_data.chords, [])


if __name__ == "__main__":
    unittest.main()

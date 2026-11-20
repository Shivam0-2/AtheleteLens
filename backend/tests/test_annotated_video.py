from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import HTTPException

import app.api.v1.router as api_v1
from app.core.config import Settings
from app.services.annotated_video import create_annotated_video


class AnnotatedVideoTests(unittest.TestCase):
    def make_source(self, root: Path) -> Path:
        source = root / "source.mp4"
        writer = cv2.VideoWriter(
            str(source), cv2.VideoWriter_fourcc(*"mp4v"), 20, (160, 90)
        )
        self.assertTrue(writer.isOpened())
        for index in range(4):
            frame = np.full((90, 160, 3), (20 * index, 70, 110), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return source

    def test_writer_preserves_dimensions_fps_and_decodes_all_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            pose = [{"index": index, "x": .25 + index * .01, "y": .5,
                     "visibility": .9} for index in range(33)]
            video_id, url = create_annotated_video(
                source, root / "outputs", [pose] * 4,
                [["Push Up", "Elbow angle: 90 deg", "Phase: TOP"]] * 4,
                expected_width=160, expected_height=90, expected_fps=20,
            )
            output = root / "outputs" / f"{video_id}.mp4"
            self.assertEqual(url, f"/api/v1/assessments/annotated-videos/{video_id}")
            self.assertTrue(output.is_file())
            capture = cv2.VideoCapture(str(output))
            count = 0
            while capture.read()[0]:
                count += 1
            self.assertEqual(count, 4)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 160)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 90)
            capture.release()

    def test_retrieval_endpoint_serves_only_generated_uuid_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_id = uuid.uuid4().hex
            (root / f"{video_id}.mp4").write_bytes(b"mp4-data")
            with patch.object(api_v1, "get_settings", return_value=Settings(root / "model.task", 100, root)):
                response = api_v1.get_annotated_video(video_id)
                self.assertEqual(response.media_type, "video/mp4")
                self.assertIn("inline", response.headers["content-disposition"])
                self.assertEqual(Path(response.path), root / f"{video_id}.mp4")
                with self.assertRaises(HTTPException):
                    api_v1.get_annotated_video("..\\secret")
                with self.assertRaises(HTTPException):
                    api_v1.get_annotated_video("0" * 32)


if __name__ == "__main__":
    unittest.main()

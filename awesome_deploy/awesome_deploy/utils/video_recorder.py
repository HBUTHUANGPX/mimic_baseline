"""Video recording utility for MuJoCo rollout visualization."""

import datetime
import os
import subprocess
from pathlib import Path

import cv2


class VideoRecorder(object):
    """Collects RGB frames and writes them to a video file on disk."""

    def __init__(
        self,
        path="./LocoMuJoCo_recordings",
        tag=None,
        video_name=None,
        fps=60,
        compress=True,
    ):
        """Initializes a new video recorder.

        Args:
            path: Directory under which recordings are stored.
            tag: Optional subdirectory name. If ``None``, a timestamp is used.
            video_name: Base filename without extension.
            fps: Output frame rate for the recorder.
            compress: Whether to re-encode the file with ffmpeg on stop.
        """
        if tag is None:
            date_time = datetime.datetime.now()
            tag = date_time.strftime("%d-%m-%Y_%H-%M-%S")

        self._path = Path(path) / tag
        self._video_name = video_name if video_name else "recording"
        self._counter = 0
        self._fps = fps
        self._compress = compress
        self._video_writer = None
        self._video_writer_path = None

    def __call__(self, frame):
        """Appends one RGB frame to the current recording.

        Args:
            frame: Frame to be added to the video with shape ``(H, W, 3)`` in
                RGB order.
        """
        assert frame is not None
        if self._video_writer is None:
            height, width = frame.shape[:2]
            self._create_video_writer(height, width)
        self._video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def _create_video_writer(self, height, width):
        """Creates the underlying OpenCV writer on the first received frame."""
        name = self._video_name
        if self._counter > 0:
            name += f"-{self._counter}.mp4"
        else:
            name += ".mp4"
        self._path.mkdir(parents=True, exist_ok=True)
        path = self._path / name
        self._video_writer_path = str(path)
        self._video_writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            self._fps,
            (width, height),
        )

    def stop(self):
        """Finalizes the current recording and optionally compresses it.

        Returns:
            Path to the recorded video file.
        """
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # GUI teardown is optional. Headless environments can safely ignore
            # this error.
            pass
        if self._video_writer is not None:
            self._video_writer.release()

        if self._compress:
            try:
                tmp_file = str(self._path / "tmp_") + self._video_name + ".mp4"
                # Re-encode with ffmpeg so the output is smaller and more
                # portable than the raw OpenCV-produced stream.
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        self._video_writer_path,
                        "-c:v",
                        "libx264",
                        "-profile:v",
                        "baseline",
                        "-preset",
                        "fast",
                        "-crf",
                        "23",
                        "-an",
                        "-r",
                        "30",
                        "-y",
                        tmp_file,
                    ],
                    stdout=subprocess.DEVNULL,
                    check=True,
                )
                os.replace(tmp_file, self._video_writer_path)
                print(
                    "Successfully compressed recorded video and saved at: ",
                    self._video_writer_path,
                )
            except subprocess.CalledProcessError as e:
                # Compression failure should not discard the original recording.
                print(f"Video compression failed: {e}")

        self._video_writer = None
        self._counter += 1
        return self._video_writer_path

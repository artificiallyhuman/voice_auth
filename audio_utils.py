"""Utility functions for recording audio and converting to MP3."""

import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

# pydub relies on ffmpeg for mp3 encoding
from pydub import AudioSegment
# We also use pydub's silence helper for trimming.
from pydub import silence as pydub_silence



RECORDINGS_DIR = Path("recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# House-keeping helpers
# ---------------------------------------------------------------------------


def cleanup_recordings_dir() -> None:
    """Delete all audio files (WAV/MP3/etc.) in *recordings/*.

    Called on application startup to purge any artefacts from previous runs.
    All embeddings are stored in the JSON DB, so keeping the raw audio is not
    necessary once processing completed.
    """

    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    for path in RECORDINGS_DIR.iterdir():
        if path.suffix.lower() in exts:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                # Best-effort: ignore files that are currently in use or
                # protected by the OS.
                pass


# ---------------------------------------------------------------------------
# High-level recorder abstraction for variable-length recordings
# ---------------------------------------------------------------------------


class Recorder:
    """Record microphone audio until *stop_and_save()* is called.

    Usage:

        rec = Recorder()
        rec.start()          # begins capturing audio from default device
        ...                  # user speaks …
        wav_path = rec.stop_and_save()  # returns path to saved WAV file
    """

    def __init__(self, samplerate: int = 16000, channels: int = 1):
        self.samplerate = samplerate
        self.channels = channels
        self._frames: List[np.ndarray] = []
        self._stream: sd.InputStream | None = None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._wav_path: Path = RECORDINGS_DIR / f"recording_{timestamp}.wav"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Begin capturing audio asynchronously (non-blocking)."""

        if self._stream is not None:
            raise RuntimeError("Recording already in progress")

        def _callback(indata, frames, time, status):  # noqa: D401, N802
            if status:
                # Print any errors reported by PortAudio to aid debugging but
                # keep recording.
                print(f"[Recorder] {status}")
            # Make a copy because *indata* is reused by the stream.
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()

    def stop_and_save(self) -> Path:
        """Stop the recording, persist to disk and return the WAV path."""

        if self._stream is None:
            raise RuntimeError("Recording was not started.")

        self._stream.stop()
        self._stream.close()
        self._stream = None

        if not self._frames:
            raise RuntimeError("No audio data captured.")

        audio_data = np.concatenate(self._frames, axis=0)
        sf.write(self._wav_path, audio_data, self.samplerate)
        return self._wav_path



def record_audio(duration_seconds: int = 5, samplerate: int = 16000) -> Path:
    """Record audio from the default microphone and return path to WAV file."""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = RECORDINGS_DIR / f"recording_{timestamp}.wav"

    print("Recording...")
    audio = sd.rec(int(duration_seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    sf.write(wav_path, audio, samplerate)
    print(f"Saved recording to {wav_path}")

    return wav_path


def convert_wav_to_mp3(wav_path: Path) -> Path:
    """Convert WAV file to MP3 using pydub (ffmpeg)."""
    mp3_path = wav_path.with_suffix(".mp3")
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")
    return mp3_path


# ---------------------------------------------------------------------------
# Silence trimming
# ---------------------------------------------------------------------------


def trim_wav_silence(
    wav_path: Path,
    *,
    silence_thresh: int = -40,
    min_silence_len: int = 300,
) -> Path:
    """Remove leading and trailing silence from *wav_path* and return new file.

    Parameters
    ----------
    wav_path
        Path to an existing WAV file.
    silence_thresh
        The volume (in dBFS) that is considered silence.  Typical values range
        from –60 (very sensitive) to –30 (conservative).  Default –40 dBFS.
    min_silence_len
        Minimum length (in ms) that must be below *silence_thresh* to be
        counted as silence.  Default 300 ms.

    Returns
    -------
    Path
        Path to the newly written, trimmed WAV file.  The original file is
        left untouched so that callers can decide whether to keep or delete
        it later.
    """

    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    audio = AudioSegment.from_wav(wav_path)

    # detect_nonsilent returns list of [start, end] in milliseconds
    nonsilent_ranges = pydub_silence.detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    if not nonsilent_ranges:
        # No speech detected – return original path to avoid breaking flow
        return wav_path

    start, _ = nonsilent_ranges[0]
    _, end = nonsilent_ranges[-1]

    trimmed_audio = audio[start:end]

    trimmed_path = wav_path.with_name(f"{wav_path.stem}_trimmed.wav")
    trimmed_audio.export(trimmed_path, format="wav")
    return trimmed_path

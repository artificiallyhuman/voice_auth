"""Local audio embedding utilities using SpeechBrain.

This module exposes a single convenience function, ``get_audio_embedding``,
that converts an *MP3* (or any file supported by *torchaudio*) into a dense
speaker-embedding vector **without relying on any external API**.  The
implementation is powered by the open-source *SpeechBrain* toolkit and its
pre-trained *ECAPA‐TDNN* model (``speechbrain/spkrec-ecapa-voxceleb``).

Why this model?

•  It is fully open source and freely available via the HuggingFace Hub.
•  It produces 192-dimensional embeddings that are state-of-the-art for
   speaker recognition / verification tasks.
•  It only requires CPU execution and runs in real-time on most laptops.

The first time you call the function, SpeechBrain automatically downloads the
model weights (~55 MB) to the local HuggingFace cache.  Subsequent calls reuse
both the cached weights **and** the instantiated classifier object so that
embedding extraction is fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torchaudio

# The SpeechBrain *EncoderClassifier* class provides a simple interface for
# obtaining fixed-dimensional embeddings from raw waveforms.
from speechbrain.pretrained import EncoderClassifier  # type: ignore

# ---------------------------------------------------------------------------
# Lazy global initialisation
# ---------------------------------------------------------------------------

_CLASSIFIER: EncoderClassifier | None = None


def _get_classifier() -> EncoderClassifier:
    """Return a singleton *EncoderClassifier* instance on CPU."""

    global _CLASSIFIER
    if _CLASSIFIER is None:
        # We explicitly place the model on CPU because VoiceGuard currently
        # ships without GPU support.  If the user does have a GPU, PyTorch can
        # transparently move the model later (``classifier.to(device)``).
        _CLASSIFIER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )
    return _CLASSIFIER


def _load_audio(path: Path) -> torch.Tensor:
    """Load *path* with *torchaudio* and return a mono 16 kHz waveform.

    The ECAPA model expects single-channel audio at 16 kHz.  This helper takes
    care of resampling and channel averaging if necessary.
    """

    # ``torchaudio`` supports most common codecs (including MP3) when it is
    # compiled with *FFmpeg* support.  On some platforms that support might be
    # missing.  If the direct load fails we convert the file to WAV on the
    # fly via *pydub* (which itself uses FFmpeg), then try again.

    try:
        waveform, sample_rate = torchaudio.load(str(path))  # shape: [channels, time]
    except Exception as exc:
        # Fallback: decode with pydub then create a tensor manually.
        try:
            from pydub import AudioSegment  # Lazy import to avoid mandatory dep

            audio = AudioSegment.from_file(path)
            sample_rate = audio.frame_rate
            # Convert AudioSegment (which stores samples as array of ints) to
            # a float32 torch tensor in the range [-1, 1].
            samples = torch.tensor(audio.get_array_of_samples(), dtype=torch.float32)
            # If stereo, interleaved samples -> reshape + average channels.
            if audio.channels > 1:
                samples = samples.view(-1, audio.channels).mean(dim=1)
            waveform = samples.unsqueeze(0) / (1 << (8 * audio.sample_width - 1))
        except Exception as exc2:  # noqa: F841
            raise RuntimeError(f"Failed to load audio file {path}: {exc}") from exc

    # Convert to mono if multi-channel.  We simply average the channels.
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    target_sr = 16000
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)

    return waveform


def get_audio_embedding(mp3_path: Path) -> List[float]:
    """Return a speaker-embedding vector for the supplied *mp3_path*.

    Parameters
    ----------
    mp3_path
        Path to an MP3 (or any audio file decodable by *torchaudio*).

    Returns
    -------
    List[float]
        A list containing 192 floating-point values representing the speaker
        embedding.
    """

    if not mp3_path.exists():
        raise FileNotFoundError(mp3_path)

    classifier = _get_classifier()

    waveform = _load_audio(mp3_path)  # shape: [1, time]

    # SpeechBrain expects shape [batch, time] **without** channel dim.  We
    # therefore squeeze any singleton channel dimension and then add the batch
    # dimension explicitly.

    if waveform.dim() == 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)  # -> [time]

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # -> [1, time]

    with torch.no_grad():
        embeddings = classifier.encode_batch(waveform)

    # ``encode_batch`` returns shape [batch, embed_dim].  We squeeze the batch
    # dimension and convert to a Python list.
    emb_vec = embeddings.squeeze(0).cpu().numpy()
    return emb_vec.tolist()

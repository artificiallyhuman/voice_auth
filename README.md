# VoiceGuard – Offline Voice Authentication Demo

VoiceGuard is a **desktop application written in Python** that demonstrates
how modern speaker–recognition models can be used to register and verify users
purely **offline** – no cloud APIs or external services are required.  The
project combines a friendly Tkinter GUI, real-time microphone recording and a
state-of-the-art *ECAPA-TDNN* speaker-embedding model provided by
[SpeechBrain](https://speechbrain.github.io/).

> **Important**  
> VoiceGuard is a **technology demo**, not a production-ready security
> solution.  The similarity threshold, anti-spoofing measures and user data
> protection have been kept deliberately simple to make the code easy to read
> and extend.


## Features

• Register new users by recording a single voice sample.  
• Verify speakers against the locally stored user database.  
• Adjustable similarity threshold and custom prompt scripts via built-in
  *Admin* panel.  
• Runs completely **offline** – embeddings are generated locally with
  SpeechBrain; user data is stored in a plain JSON file.  
• Cross-platform (Windows, macOS, Linux) as long as the system provides an
  input audio device and Python ≥ 3.10.


## Quick start

1. **Clone the repository**

   ```bash
   git clone https://github.com/artificiallyhuman/voice_auth.git
   cd voice_auth
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install the dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   The repository already contains the ~55 MB pre-trained ECAPA model inside
   `pretrained_models/`, so no additional download is required on first run.

4. **Launch the GUI**

   ```bash
   python main.py
   ```

   You should see the VoiceGuard window with three tabs: *Register* (default),
   *Verify* and *Admin*.


## Usage workflow

1. **Register**  – Fill in the user’s first name, last name and date of birth,
   press **Next**, read the displayed script out loud and press **Done**.  The
   application trims silence, creates a 192-dimensional embedding and stores
   it along with the metadata in `users.json`.

2. **Verify**  – Select the *Verify* tab, hit **Ready**, read the (potentially
   different) prompt and press **Done** again.  VoiceGuard compares the new
   embedding against all registered users using cosine similarity and either
   displays a match or a *No match* warning.

3. **Admin**  – Fine-tune the similarity threshold (0–1) and change the prompt
   scripts that are shown during registration and verification.  Settings are
   persisted in `config.json`.


## Project structure

```
voice_auth/
├── audio_embedding.py   # SpeechBrain wrapper
├── audio_utils.py       # Recording and silence-trimming helpers (optional
|                         WAV→MP3 conversion utilities are also included)
├── config_utils.py      # JSON config loader / saver
├── db.py                # Lightweight JSON persistence replacing SQL
├── main.py              # Tkinter GUI application entry-point
├── requirements.txt     # Python dependencies
├── pretrained_models/   # Cached ECAPA model (auto-created)
├── recordings/          # Temporary audio files (auto-purged on start-up)
├── users.json           # VoiceGuard user database (auto-created)
└── config.json          # App settings (auto-created)
```


## Dependencies & prerequisites

Mandatory Python packages are listed in `requirements.txt`.  In addition you
need:

• **FFmpeg** – Only needed if you plan to use the optional MP3 helpers in
  `audio_utils.py` or if `torchaudio` on your platform lacks native MP3
  support.  Make sure the `ffmpeg` binary is available on your `PATH` (e.g.
  `brew install ffmpeg` on macOS).  
• A working **microphone** and a recent version of **PortAudio** (bundled with
  the `sounddevice` Python package for most platforms).


## Configuration files

• `config.json` – Stores global app settings.  Created automatically with
  sensible defaults on first launch.  
• `users.json` – Holds an array of registered users and their embeddings.

Both files are **human readable** and can be edited manually, but using the
*Admin* panel is safer.


## Adjusting the similarity threshold

The default threshold of **0.80** works reasonably well for the ECAPA
embeddings but may require tweaking depending on microphone quality and
environmental noise:

• **Higher threshold** → fewer false positives, more false rejections.  
• **Lower threshold** → more tolerant matching, higher risk of impersonation.

Experiment and find the sweet spot for your use-case in the *Admin* tab.


## Known limitations

• No anti-spoofing / liveness detection – a high-quality recording of the user
  could bypass verification.  
• Single sample enrollment – adding multiple recordings per user would improve
  robustness.  
• GUI blocks for a second or two while the ECAPA model is first loaded into
  memory.  (The weights ship with the repo, so there is no network delay.)


## Contributing

Pull requests are welcome!  Feel free to open issues for suggestions or bug
reports.  To set up a development environment simply create a virtual
environment and install the dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If you use *pre-commit* in your own workflow you can of course run it locally,
but the project does not require it.


## License

This project is released under the MIT License – see `LICENSE` for details.

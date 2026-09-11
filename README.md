<div align="center">

# Study Guardian

**A real-time webcam focus watchdog powered by YOLOv8.**

Catches you reaching for your phone and calls you out  on screen and out loud.

Python 3.8+ · YOLOv8 · OpenCV · MIT License · Windows, macOS and Linux

</div>

---

## Overview

Study Guardian is a desktop accountability tool for anyone who studies with a phone within arm's reach. It watches your desk through the webcam and reacts to what it sees in real time.

Pick up your phone and the interface flashes red while a voice tells you to get back to work. Pick up a book and it turns green with a bit of encouragement. Put both down and it settles into a quiet waiting state.

Everything runs locally on your machine. No frames are recorded, stored or sent anywhere.

## Features

- **Real-time object detection** using a pretrained YOLOv8 model, running at interactive frame rates on CPU.
- **Three-state feedback system** with distinct colour schemes, status pills and bounding-box overlays.
- **Offline text-to-speech** through pyttsx3, with a background worker thread so speech never blocks the video loop.
- **Hysteresis-based state machine** that holds each state briefly after the last detection, preventing visual flicker from dropped frames.
- **Remote-as-phone heuristic**, since a phone held at an angle is frequently classified as a remote by COCO-trained models.
- **Live debug overlay** showing raw class names and confidence scores, toggleable at runtime.
- **Fully configurable** through command-line arguments and a single block of tunable constants.

## How it works

### Detection pipeline

Frames are captured through OpenCV and mirrored horizontally so the preview behaves like a mirror. Every *n*-th frame (2 by default) is passed to YOLOv8 with a low confidence floor, and the results are filtered down to three COCO classes: `cell phone`, `book` and `remote`. Detections below their per-class threshold are discarded, and the survivors update a pair of "last seen" timestamps.

Running detection on a subset of frames keeps the interface responsive while leaving inference cost roughly proportional to `1 / DETECT_EVERY`.

### State machine

| State | Trigger condition | Visual treatment | Voice |
|---|---|---|---|
| `phone` | Phone or remote seen within the last 3000 ms | Pulsing red background, red border, `DISTRACTED` pill | Warning message |
| `book` | Book seen within the last 1500 ms and no recent phone | Green background, green border, `FOCUSED` pill | Encouragement |
| `neutral` | Nothing seen for 4000 ms | Dark background, `WAITING...` pill | Silent |

Phone detections take priority over books, so a phone held over an open textbook still triggers the warning. Speech fires only on state transitions, never on every frame, and the speech queue drains stale messages so the audio cannot fall behind the visuals.

## Requirements

- Python 3.8 or newer
- A working webcam
- A desktop session — the application opens a GUI window and will not run headless
- Roughly 100 MB of disk space for the model weights

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/study-guardian.git
cd study-guardian

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The YOLOv8 weights are downloaded automatically on first launch, so the initial run requires an internet connection and takes slightly longer than subsequent ones.

**Linux users** also need a speech backend for pyttsx3:

```bash
sudo apt install espeak
```

## Usage

```bash
python study_guardian.py
```

### Command-line options

| Flag | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera index. Use `1` or higher if the wrong device opens. |
| `--model` | `yolov8n.pt` | YOLO weights. `yolov8s.pt` is more accurate but slower. |
| `--name` | `Tan` | Name used in the on-screen and spoken messages. |
| `--no-voice` | *off* | Run without text-to-speech. |
| `--width` | `1280` | Requested capture width. |
| `--height` | `720` | Requested capture height. |

```bash
# Higher accuracy, second camera, custom name
python study_guardian.py --model yolov8s.pt --camera 1 --name Alex

# Silent mode at a lower resolution
python study_guardian.py --no-voice --width 960 --height 540
```

### Keyboard controls

| Key | Action |
|:---:|---|
| <kbd>Q</kbd> / <kbd>Esc</kbd> | Quit |
| <kbd>M</kbd> | Mute or unmute the voice |
| <kbd>D</kbd> | Toggle the debug overlay |

## Configuration

Sensitivity and timing are controlled by the constants at the top of `study_guardian.py`:

```python
PHONE_SCORE  = 0.22    # confidence required to register a phone
BOOK_SCORE   = 0.40    # confidence required to register a book
REMOTE_SCORE = 0.35    # remotes are treated as phones

PHONE_HOLD_MS    = 3000   # how long the phone state persists
BOOK_HOLD_MS     = 1500   # how long the book state persists
NEUTRAL_AFTER_MS = 4000   # idle time before returning to neutral

DETECT_EVERY = 2          # run inference every Nth frame
```

The phone threshold is set deliberately low: phones occupy few pixels and are usually partly occluded by a hand, so a strict threshold misses most of them. Raise `PHONE_SCORE` if you see false alarms, lower `BOOK_SCORE` if books go unnoticed, and raise `DETECT_EVERY` to trade reaction time for a smoother frame rate.

## Project structure

```
study-guardian/
├── study_guardian.py     # application entry point
├── requirements.txt      # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```

The script itself is organised into three parts: a `Speaker` class wrapping threaded text-to-speech, a `Detector` class wrapping YOLOv8 inference and class filtering, and a set of drawing helpers that compose each output frame.

## Troubleshooting

<details>
<summary><strong>"Couldn't open camera 0"</strong></summary>

Another application is likely holding the webcam, or the device index is wrong. Close any other app using the camera, then try `--camera 1`. On macOS, make sure your terminal has camera permission under System Settings → Privacy & Security → Camera.
</details>

<details>
<summary><strong>No voice output</strong></summary>

The application prints a notice and continues silently when pyttsx3 or the system speech engine is unavailable. Confirm pyttsx3 is installed, install `espeak` on Linux, and check you have not muted with <kbd>M</kbd>.
</details>

<details>
<summary><strong>Low frame rate</strong></summary>

Stay on `yolov8n.pt`, reduce the capture resolution with `--width 960 --height 540`, or increase `DETECT_EVERY` to 3 or 4.
</details>

<details>
<summary><strong>Phone is not being detected</strong></summary>

Lighting matters a great deal. Hold the phone facing the camera rather than edge-on, and lower `PHONE_SCORE` if detections are still unreliable in your setup.
</details>

## Limitations

- Accuracy is bounded by the pretrained COCO model. Poor lighting, small objects and unusual angles all reduce reliability.
- Notebooks, folders and laptops are occasionally classified as books.
- The application requires continuous camera access while running, though no video data leaves the machine.
- CPU inference on older hardware may not sustain a smooth frame rate at 720p.

## Roadmap

- [ ] Session statistics: time focused versus time distracted
- [ ] Configuration file so thresholds can be changed without editing source
- [ ] Optional logging and an end-of-session summary
- [ ] Custom-trained weights for better phone recognition at desk distance
- [ ] Packaged executable for one-click use

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection model
- [OpenCV](https://opencv.org/) for capture and rendering
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for offline speech synthesis

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
Built by <strong>Md Tanvir Ahmmed</strong> — if this helped you stay off your phone, consider leaving a star.
</div>

# VOXLENS 👓🎙️

**Smart glasses that turn speech into text, in real time, for the deaf and hard of hearing.**

VOXLENS listens to the people around you, converts their speech to text, and shows it on a small OLED display so conversations become readable instantly.

🏆 Winner, TechExpo 2025

---

## How it works

```
 Microphone ──► Speech Recognition (Python) ──► Serial / USB ──► ESP32 ──► OLED Display
```

1. A microphone captures speech.
2. The Python script converts speech to text using Google Speech Recognition.
3. The text is sent to the ESP32 over serial.
4. The ESP32 renders the text on an SSD1306 OLED display.

## Features

- Real-time speech-to-text
- Compact OLED output, suitable for wearable use
- Ambient noise calibration for better accuracy
- Multi-language support (English-India by default, Hindi supported via `hi-IN`)
- Simple, hackable codebase

## Hardware

| Component | Details |
|---|---|
| Microcontroller | ESP32 |
| Display | SSD1306 OLED, 128x64, I2C |
| Microphone | Laptop / USB mic (INMP441 I2S mic planned) |
| Connection | USB serial |

### Wiring (OLED to ESP32)

| OLED | ESP32 |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | GPIO 21 |
| SCL | GPIO 22 |

## Software requirements

- Python 3.8+
- Arduino IDE (with ESP32 board support)
- Arduino libraries: `Adafruit SSD1306`, `Adafruit GFX`

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/voxlens-stt.git
cd voxlens-stt
```

### 2. Flash the ESP32

1. Open `voxlens_esp32/voxlens_esp32.ino` in Arduino IDE.
2. Select your ESP32 board and COM port.
3. Upload.

### 3. Run the speech-to-text script

```bash
pip install -r requirements.txt
```

Open `voxlens_stt.py` and set `PORT` to your ESP32's port (e.g. `COM3` on Windows, `/dev/ttyUSB0` on Linux). Then:

```bash
python voxlens_stt.py
```

Speak, and your words appear on the OLED.

## Media recognition (audio / video files)

VOXLENS can also transcribe recorded media such as lectures, videos, and audio clips, and stream the text to the OLED.

Requires [ffmpeg](https://ffmpeg.org) installed and available in PATH.

```bash
# Transcribe and show on OLED
python voxlens_media.py lecture.mp4 --port COM3

# Hindi audio
python voxlens_media.py song.mp3 --lang hi-IN

# Only transcript, no ESP32
python voxlens_media.py video.mp4 --no-serial --save transcript.txt
```

| Flag | Description |
|---|---|
| `--port` | ESP32 serial port |
| `--lang` | Recognition language (`en-IN`, `hi-IN`, `en-US`) |
| `--chunk` | Chunk size in seconds (default 8) |
| `--no-serial` | Skip ESP32, print to terminal only |
| `--save` | Save timestamped transcript to a `.txt` file |

## Configuration

| Setting | File | Default | Description |
|---|---|---|---|
| `PORT` | `voxlens_stt.py` | `COM3` | Serial port of the ESP32 |
| `BAUD` | both | `115200` | Serial baud rate |
| `LANG` | `voxlens_stt.py` | `en-IN` | Recognition language |
| `setTextSize` | `.ino` | `2` | Use `1` to fit more text |

## Project structure

```
voxlens-stt/
├── voxlens_stt.py
├── voxlens_esp32/
│   └── voxlens_esp32.ino
├── requirements.txt
└── README.md
```

## Troubleshooting

- **Nothing on OLED:** check I2C address (`0x3C` vs `0x3D`) and SDA/SCL wiring.
- **Serial error:** close Arduino Serial Monitor, since it blocks the port.
- **"Samajh nahi aaya" repeatedly:** reduce background noise or move the mic closer.
- **PyAudio install fails:** on Windows use `pip install pipwin && pipwin install pyaudio`; on Linux install `portaudio19-dev` first.

## Limitations

- Needs internet for Google Speech Recognition.
- OLED shows roughly the last 40 characters at text size 2.
- Currently uses a laptop-side mic and processing.

## Roadmap

- [ ] I2S microphone (INMP441) directly on ESP32
- [ ] WiFi / BLE transmission instead of USB serial
- [ ] Offline speech recognition (Vosk / Whisper)
- [ ] Scrolling text and Devanagari font support
- [ ] Battery-powered, glasses-mounted enclosure

## Author

**Pragya**
Co-Founder & COO, VOIKES Technologies
B.Tech CSE, PSIT Kanpur

## License

MIT License. Free to use, modify, and share with credit.

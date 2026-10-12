"""
VOXLENS - Speech to Text (PC / Laptop side)
Mic se speech sunta hai -> text me convert karta hai -> ESP32 ko serial se bhejta hai

Install:
    pip install SpeechRecognition pyaudio pyserial
"""

import time
import serial
import speech_recognition as sr

PORT = "COM3"          # Linux/Mac: "/dev/ttyUSB0" ya "/dev/cu.usbserial-xxxx"
BAUD = 115200
LANG = "en-IN"         # Hindi ke liye "hi-IN"

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # ESP32 reset hone ka wait

recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = True
mic = sr.Microphone()

print("Calibrating noise...")
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)
print("VOXLENS ready. Bolo...")

while True:
    try:
        with mic as source:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=6)

        text = recognizer.recognize_google(audio, language=LANG)
        print("Heard:", text)
        ser.write((text + "\n").encode("utf-8"))

    except sr.WaitTimeoutError:
        pass
    except sr.UnknownValueError:
        print("Samajh nahi aaya...")
    except sr.RequestError as e:
        print("Speech API error:", e)
    except KeyboardInterrupt:
        break

ser.close()

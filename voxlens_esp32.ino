/*
  VOXLENS - ESP32 side
  Serial se text receive karta hai aur SSD1306 OLED (128x64) pe dikhata hai

  Wiring (I2C OLED):
    OLED VCC -> 3.3V
    OLED GND -> GND
    OLED SDA -> GPIO 21
    OLED SCL -> GPIO 22

  Libraries: Adafruit SSD1306, Adafruit GFX
*/

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

String incoming = "";

void showText(const String &msg) {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(2);        // chhota chahiye to 1 kar do
  display.setTextColor(SSD1306_WHITE);
  display.setTextWrap(true);
  display.println(msg);
  display.display();
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    while (true);  // OLED nahi mila
  }

  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 20);
  display.println("VOXLENS");
  display.display();
  delay(1500);
  showText("Listening...");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      incoming.trim();
      if (incoming.length() > 0) {
        // OLED pe sirf last ~40 chars fit hote hain (size 2)
        if (incoming.length() > 40) {
          incoming = incoming.substring(incoming.length() - 40);
        }
        showText(incoming);
      }
      incoming = "";
    } else {
      incoming += c;
    }
  }
}

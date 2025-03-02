#include <Arduino.h>
#include <ESP32Servo.h>  // Use <Servo.h> if using the standard library

// Define the servo object and pins
Servo myServo;
const int servoPin = 15; // Servo connected to GPIO 15
const int pirPin = 14;   // PIR sensor connected to GPIO 14
bool pir_triggered = false; // Flag to indicate motion detection

void setup() {
  Serial.begin(115200);  // Initialize serial communication at 115200 baud
  myServo.attach(servoPin);  // Attach the servo to the specified pin
  pinMode(pirPin, INPUT);    // Set PIR pin as input
  myServo.write(110);         // Set default position to 110 degrees
  Serial.println("Servo set to default position: 110°");
}

void loop() {
  // Check PIR sensor status
  if (digitalRead(pirPin) == HIGH) {
    pir_triggered = true;
    // Serial.println("MOTION_DETECTED");
    Serial.println("PIR Flag: True");
  } else {
    pir_triggered = false;
    Serial.println("PIR Flag: False");
  }

  // Check if data is available on the Serial port
  if (Serial.available() > 0) {
    // Read the incoming command until newline
    String command = Serial.readStringUntil('\n');
    command.trim();  // Remove any leading/trailing whitespace

    // Process the command and move servo accordingly
    if (command == "recyclable") {
      myServo.write(60);
      Serial.println("Command 'recyclable' received. Servo moved to 60°.");
      delay(1000);  // Wait for 1 second
      myServo.write(110);
    } else if (command == "non_recyclable") {
      myServo.write(180);
      Serial.println("Command 'non_recyclable' received. Servo moved to 180°.");
      delay(1000);  // Wait for 1 second
      myServo.write(110);
    }
  }
}

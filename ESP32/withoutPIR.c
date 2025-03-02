#include <ESP32Servo.h>

static const int servoPin = 13;
Servo servo1;

const int originPosition = 90; // Set the neutral (horizontal) position
const int offset = 70;         // Degrees to move left or right

void setup() {
  Serial.begin(115200);
  servo1.attach(servoPin);
  servo1.write(originPosition); // Ensure servo starts at the neutral position
  delay(500);
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); // Read input until newline
    command.trim(); // Remove any leading/trailing spaces

    if (command == "recyclable") {
      moveServo(originPosition, originPosition + offset); // Move right
      delay(1000); // Small delay before returning
      moveServo(originPosition + offset, originPosition); // Return to origin
    } 
    else if (command == "non-recyclable") {
      moveServo(originPosition, originPosition - offset); // Move left
      delay(1000); // Small delay before returning
      moveServo(originPosition - offset, originPosition); // Return to origin
    }
  }
}

// Function to smoothly move the servo
void moveServo(int start, int end) {
  int step = (start < end) ? 1 : -1; // Determine movement direction
  for (int pos = start; pos != end; pos += step) {
    servo1.write(pos);
    Serial.println(pos);
    delay(2); // Adjust for smooth movement
  }
  servo1.write(end); // Ensure final position is reached
}

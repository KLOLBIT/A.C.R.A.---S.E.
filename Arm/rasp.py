#!/usr/bin/env python3
import socket
import struct
import time
from adafruit_servokit import ServoKit

# Initialize ServoKit for Waveshare HAT (16 channels)
kit = ServoKit(channels=16)

# Set all five finger servos to 0° at startup
for i in range(5):
    kit.servo[i].angle = 0
# allow servos to move to 0°
time.sleep(0.5)

# UDP server settings
sock.bind((HOST, PORT))

print(f"Listening for servo angles on UDP port {PORT}...")

try:
    while True:
        data, addr = sock.recvfrom(5)
        if not data:
            continue

        # Unpack five unsigned bytes
        angles = struct.unpack('5B', data)
        print(f"Received {angles} from {addr}")

        # Set each channel (0–4) to the corresponding angle
        for i, ang in enumerate(angles):
            kit.servo[i].angle = ang  # channels 0–4 wired to five servos

        # small delay for servo movement
        time.sleep(0.02)

except KeyboardInterrupt:
    print("Shutting down.")

finally:
    sock.close()

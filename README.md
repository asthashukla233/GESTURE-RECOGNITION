✋ Improved Gesture Control System
Control your volume and brightness using just hand gestures via your webcam!
This system uses MediaPipe, OpenCV, and OS-level commands to provide cross-platform support for gesture-based media control.

📦 Features
✌️ Peace Sign (Index + Middle fingers) → Volume Control

🤏 Pinch Gesture (Thumb + Index finger) → Brightness Control

👍 Thumbs Up Gesture → Alternative Volume Control

💻 Works on Windows, macOS, and Linux with platform-specific enhancements

🔄 Reset to default values using the r key

🛠 Requirements
Install the required packages:

bash
Copy
Edit
pip install opencv-python mediapipe numpy
Windows Only:
bash
Copy
Edit
pip install pycaw wmi
For better volume control, download nircmd.exe from NirSoft and place it in your system PATH.

Linux:
Ensure you have at least one of these installed:

amixer

pactl

pamixer

🚀 How to Run
bash
Copy
Edit
python gesture.py
Make sure your webcam is accessible and functioning.

🔑 Controls & Usage
Gesture	Function
Peace Sign	Volume Adjustment
Pinch (Thumb + Index)	Brightness Control
Thumbs Up	Alternative Volume
q key	Quit Program
r key	Reset to 50% values

🧠 Behind the Scenes
Uses MediaPipe Hands to detect hand landmarks in real-time.

Dynamically calculates distances between finger tips to interpret gestures.

Controls system volume/brightness using platform-native methods.

💡 Notes
Frame is flipped horizontally for intuitive interaction.

Throttling is applied to prevent frequent system calls (updates every 50ms).

Gesture drawing and finger status help with real-time feedback and debugging.

📋 Troubleshooting
Camera not opening? Make sure no other app is using the webcam.

Volume not changing on Windows? Try placing nircmd.exe in the script directory or your system path.

Brightness not working? Windows users may need to run the script as Administrator.


import cv2
import mediapipe as mp
import numpy as np
import math
import platform
import subprocess
import time

class ImprovedGestureControl:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # System info
        self.os_type = platform.system()
        print(f"Detected OS: {self.os_type}")
        
        # Control values
        self.current_volume = 50
        self.current_brightness = 50
        self.last_update_time = 0
        
        # Gesture detection parameters
        self.min_distance = 20
        self.max_distance = 150
        
        # Initialize system controls
        self.init_system_controls()
    
    def init_system_controls(self):
        """Initialize system-specific controls with better volume support"""
        print("Initializing system controls...")
        
        if self.os_type == "Windows":
            try:
                # Try multiple Windows volume methods
                self.init_windows_volume()
            except Exception as e:
                print(f"❌ Windows volume control failed: {e}")
                self.volume_interface = None
                
        elif self.os_type == "Darwin":  # macOS
            print("✅ macOS controls ready")
            
        elif self.os_type == "Linux":
            self.init_linux_audio()
    
    def init_windows_volume(self):
        """Initialize Windows volume control with multiple fallback methods"""
        try:
            # Method 1: Try pycaw
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
            self.volume_method = "pycaw"
            print("✅ Windows volume control (pycaw) initialized")
            
        except ImportError:
            print("⚠️  pycaw not available, using nircmd fallback")
            self.volume_interface = None
            self.volume_method = "nircmd"
            
        except Exception as e:
            print(f"⚠️  pycaw failed ({e}), using nircmd fallback")
            self.volume_interface = None
            self.volume_method = "nircmd"
    
    def init_linux_audio(self):
        """Initialize Linux audio with better detection"""
        self.linux_audio_method = None
        
        # Test amixer
        try:
            result = subprocess.run(["amixer", "get", "Master"], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                self.linux_audio_method = "amixer"
                print("✅ Linux audio control (amixer) ready")
                return
        except:
            pass
        
        # Test pactl
        try:
            result = subprocess.run(["pactl", "info"], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                self.linux_audio_method = "pactl"
                print("✅ Linux audio control (pactl) ready")
                return
        except:
            pass
        
        # Test pamixer
        try:
            result = subprocess.run(["pamixer", "--get-volume"], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                self.linux_audio_method = "pamixer"
                print("✅ Linux audio control (pamixer) ready")
                return
        except:
            pass
        
        print("❌ No Linux audio control found")
    
    def set_volume(self, volume):
        """Improved volume control with multiple methods"""
        volume = max(0, min(100, volume))
        
        try:
            if self.os_type == "Windows":
                if hasattr(self, 'volume_method') and self.volume_method == "pycaw" and self.volume_interface:
                    # Method 1: pycaw
                    self.volume_interface.SetMasterScalarVolume(volume / 100.0, None)
                    print(f"🔊 Volume (pycaw): {volume}%")
                else:
                    # Method 2: nircmd (download from nirsoft.net)
                    try:
                        subprocess.run(["nircmd.exe", "setsysvolume", str(int(volume * 655.35))], 
                                     capture_output=True, timeout=1)
                        print(f"🔊 Volume (nircmd): {volume}%")
                    except:
                        # Method 3: PowerShell
                        ps_cmd = f"(New-Object -comObject WScript.Shell).SendKeys([char]175)"  # Volume up key
                        subprocess.run(["powershell", "-Command", ps_cmd], 
                                     capture_output=True, timeout=2)
                        print(f"🔊 Volume (PowerShell): {volume}%")
                
            elif self.os_type == "Darwin":  # macOS
                subprocess.run(["osascript", "-e", f"set volume output volume {volume}"], 
                             capture_output=True, timeout=2)
                print(f"🔊 Volume: {volume}%")
                
            elif self.os_type == "Linux":
                if self.linux_audio_method == "amixer":
                    subprocess.run(["amixer", "set", "Master", f"{volume}%"], 
                                 capture_output=True, timeout=2)
                elif self.linux_audio_method == "pactl":
                    subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{volume}%"], 
                                 capture_output=True, timeout=2)
                elif self.linux_audio_method == "pamixer":
                    subprocess.run(["pamixer", "--set-volume", str(volume)], 
                                 capture_output=True, timeout=2)
                print(f"🔊 Volume: {volume}%")
                    
        except Exception as e:
            print(f"❌ Volume control failed: {e}")
    
    def set_brightness(self, brightness):
        """Improved brightness control"""
        brightness = max(0, min(100, brightness))
        
        try:
            if self.os_type == "Windows":
                try:
                    # Method 1: WMI
                    import wmi
                    c = wmi.WMI(namespace='wmi')
                    methods = c.WmiMonitorBrightnessMethods()[0]
                    methods.WmiSetBrightness(brightness, 0)
                    print(f"💡 Brightness: {brightness}%")
                except:
                    # Method 2: PowerShell
                    ps_cmd = f'(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{brightness})'
                    subprocess.run(["powershell", "-Command", ps_cmd], 
                                 capture_output=True, timeout=3)
                    print(f"💡 Brightness: {brightness}%")
                
            elif self.os_type == "Darwin":  # macOS
                brightness_val = brightness / 100.0
                subprocess.run(["osascript", "-e", 
                              f"tell application \"System Events\" to set brightness of every display to {brightness_val}"], 
                             capture_output=True, timeout=2)
                print(f"💡 Brightness: {brightness}%")
                
            elif self.os_type == "Linux":
                # Method 1: xrandr
                brightness_val = brightness / 100.0
                result = subprocess.run(["xrandr", "--listmonitors"], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[1:]:
                        if line.strip():
                            monitor = line.split()[-1]
                            subprocess.run(["xrandr", "--output", monitor, 
                                          "--brightness", str(brightness_val)], 
                                         capture_output=True, timeout=2)
                            break
                print(f"💡 Brightness: {brightness}%")
                            
        except Exception as e:
            print(f"❌ Brightness control failed: {e}")
    
    def get_distance(self, p1, p2):
        """Calculate distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def detect_gesture(self, landmarks, frame_shape):
        """Detect gestures with new, more intuitive gestures"""
        h, w = frame_shape[:2]
        
        # Get key landmarks
        thumb_tip = np.array([landmarks[4].x * w, landmarks[4].y * h])
        index_tip = np.array([landmarks[8].x * w, landmarks[8].y * h])
        middle_tip = np.array([landmarks[12].x * w, landmarks[12].y * h])
        ring_tip = np.array([landmarks[16].x * w, landmarks[16].y * h])
        pinky_tip = np.array([landmarks[20].x * w, landmarks[20].y * h])
        
        # Get PIP joints for finger detection
        thumb_ip = np.array([landmarks[3].x * w, landmarks[3].y * h])
        index_pip = np.array([landmarks[6].x * w, landmarks[6].y * h])
        middle_pip = np.array([landmarks[10].x * w, landmarks[10].y * h])
        ring_pip = np.array([landmarks[14].x * w, landmarks[14].y * h])
        pinky_pip = np.array([landmarks[18].x * w, landmarks[18].y * h])
        
        # Detect which fingers are up
        fingers_up = []
        
        # Thumb (check x-axis)
        if landmarks[4].x > landmarks[3].x:  # Right hand
            fingers_up.append(1)
        else:
            fingers_up.append(0)
        
        # Other fingers (check y-axis)
        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        finger_pips = [index_pip, middle_pip, ring_pip, pinky_pip]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if tip[1] < pip[1]:  # Tip is above pip
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        gesture_info = {
            'type': None,
            'value': None,
            'fingers': fingers_up,
            'points': None
        }
        
        # NEW GESTURES:
        
        # 1. Volume Control: Peace sign (Index + Middle fingers up)
        if fingers_up == [0, 1, 1, 0, 0]:  # Only index and middle up
            distance = self.get_distance(index_tip, middle_tip)
            if distance >= self.min_distance:
                volume = np.interp(distance, [self.min_distance, self.max_distance], [0, 100])
                gesture_info['type'] = 'volume'
                gesture_info['value'] = int(volume)
                gesture_info['points'] = (index_tip.astype(int), middle_tip.astype(int))
        
        # 2. Brightness Control: Pinch gesture (Thumb + Index)
        elif fingers_up == [1, 1, 0, 0, 0]:  # Thumb and index up
            distance = self.get_distance(thumb_tip, index_tip)
            if distance >= self.min_distance:
                brightness = np.interp(distance, [self.min_distance, self.max_distance], [0, 100])
                gesture_info['type'] = 'brightness'
                gesture_info['value'] = int(brightness)
                gesture_info['points'] = (thumb_tip.astype(int), index_tip.astype(int))
        
        # 3. Alternative Volume: Thumbs up with distance from palm center
        elif fingers_up == [1, 0, 0, 0, 0]:  # Only thumb up
            # Use thumb tip distance from wrist
            wrist = np.array([landmarks[0].x * w, landmarks[0].y * h])
            distance = self.get_distance(thumb_tip, wrist)
            if distance >= 50:  # Minimum distance from wrist
                volume = np.interp(distance, [50, 150], [0, 100])
                gesture_info['type'] = 'volume_alt'
                gesture_info['value'] = int(volume)
                gesture_info['points'] = (thumb_tip.astype(int), wrist.astype(int))
        
        return gesture_info
    
    def run(self):
        """Main control loop with improved UI"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        print("\n🎮 IMPROVED GESTURE CONTROL!")
        print("=" * 50)
        print("✌️  Peace Sign (Index+Middle) = Volume Control")
        print("🤏 Pinch (Thumb+Index) = Brightness Control")
        print("👍 Thumbs Up Distance = Alternative Volume")
        print("Press 'q' to quit | 'r' to reset")
        print("=" * 50)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Create a semi-transparent overlay for better UI
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Display current values
            cv2.putText(frame, f"🔊 Volume: {self.current_volume}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"💡 Brightness: {self.current_brightness}%", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            current_gesture = "None"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    # Detect gesture
                    gesture = self.detect_gesture(hand_landmarks.landmark, frame.shape)
                    
                    if gesture['type']:
                        current_time = time.time()
                        
                        # Update controls with throttling
                        if current_time - self.last_update_time > 0.05:  # 50ms throttle
                            
                            if gesture['type'] == 'volume':
                                self.current_volume = gesture['value']
                                self.set_volume(self.current_volume)
                                self.last_update_time = current_time
                                current_gesture = f"✌️ Peace Sign - Volume: {self.current_volume}%"
                                
                                # Visual feedback
                                p1, p2 = gesture['points']
                                cv2.line(frame, tuple(p1), tuple(p2), (0, 255, 0), 5)
                                cv2.circle(frame, tuple(p1), 10, (0, 255, 0), -1)
                                cv2.circle(frame, tuple(p2), 10, (0, 255, 0), -1)
                            
                            elif gesture['type'] == 'brightness':
                                self.current_brightness = gesture['value']
                                self.set_brightness(self.current_brightness)
                                self.last_update_time = current_time
                                current_gesture = f"🤏 Pinch - Brightness: {self.current_brightness}%"
                                
                                # Visual feedback
                                p1, p2 = gesture['points']
                                cv2.line(frame, tuple(p1), tuple(p2), (255, 0, 255), 5)
                                cv2.circle(frame, tuple(p1), 10, (255, 0, 255), -1)
                                cv2.circle(frame, tuple(p2), 10, (255, 0, 255), -1)
                            
                            elif gesture['type'] == 'volume_alt':
                                self.current_volume = gesture['value']
                                self.set_volume(self.current_volume)
                                self.last_update_time = current_time
                                current_gesture = f"👍 Thumbs Up - Volume: {self.current_volume}%"
                                
                                # Visual feedback
                                p1, p2 = gesture['points']
                                cv2.line(frame, tuple(p1), tuple(p2), (0, 255, 255), 3)
                                cv2.circle(frame, tuple(p1), 10, (0, 255, 255), -1)
                        
                        # Show finger status for debugging
                        finger_status = "".join(['🟢' if f else '🔴' for f in gesture['fingers']])
                        cv2.putText(frame, f"Fingers: {finger_status}", 
                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show current gesture
            cv2.putText(frame, f"Gesture: {current_gesture}", 
                       (10, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Instructions
            instructions = [
                "✌️ Peace Sign = Volume | 🤏 Pinch = Brightness | 👍 Thumbs Up = Alt Volume",
                "Press 'q' to quit | 'r' to reset values"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, 
                           (10, h-50+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Improved Gesture Control', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.current_volume = 50
                self.current_brightness = 50
                self.set_volume(50)
                self.set_brightness(50)
                print("🔄 Reset to 50%")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Gesture control stopped")

if __name__ == "__main__":
    try:
        print("🚀 Starting Improved Gesture Control...")
        controller = ImprovedGestureControl()
        controller.run()
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n📋 Required packages:")
        print("pip install opencv-python mediapipe numpy")
        if platform.system() == "Windows":
            print("pip install pycaw wmi")
            print("\nFor better Windows volume control, download nircmd.exe from nirsoft.net")
        elif platform.system() == "Linux":
            print("Make sure you have: amixer, pactl, or pamixer installed")
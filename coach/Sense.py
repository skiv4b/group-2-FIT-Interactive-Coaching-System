import cv2
import mediapipe as mp
import math
import numpy as np


# Sense Component: Detect joints using the camera
# Things you need to improve: Make the skeleton tracking smoother and robust to errors.
class Sense:

    def __init__(self):
        # Initialize the Mediapipe Pose object to track joints
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose.Pose()

        # used later for having a moving avergage
        self.angle_window = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.previous_angle = -1

        # NEW Initialize MediaPipe Hands for hand detection
        # To enable hand tracking and gesture recognition for the game
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Process video frames (not static images)
            max_num_hands=2,  # Detect up to 2 hands
            min_detection_confidence=0.5,  # Minimum confidence to consider detection valid
            min_tracking_confidence=0.5  # Minimum confidence to continue tracking
        )

    def detect_joints(self, frame):
        results = self.mp_pose.process(frame)
        return results if results else None

    def calculate_angle(self, joint1, joint2, joint3):
        """
        Calculates the angle between three joints.

        Parameters:
        - joint1: Tuple of (x, y) for the first joint (e.g., shoulder)
        - joint2: Tuple of (x, y) for the middle joint (e.g., elbow)
        - joint3: Tuple of (x, y) for the last joint (e.g., wrist)

        Returns:
        - Angle in degrees between the three joints
        """
        # Calculate vectors
        vector1 = [joint1[0] - joint2[0], joint1[1] - joint2[1]]
        vector2 = [joint3[0] - joint2[0], joint3[1] - joint2[1]]

        # Calculate the dot product and magnitude of the vectors
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        # Calculate the angle in radians and convert to degrees
        angle = math.acos(dot_product / (magnitude1 * magnitude2))

        # get a moving average
        self.angle_window.pop(0)
        self.angle_window.append(angle)
        # Use np.convolve to calculate the moving average
        window_size = 10
        angle_mvg = np.convolve(np.asarray(self.angle_window), np.ones(window_size) / window_size, mode='valid')

        return math.degrees(angle_mvg)

    def extract_joint_coordinates(self, landmarks, joint):
        """
        Extracts the (x, y) coordinates of a specific joint.

        Parameters:
        - landmarks: The list of pose landmarks from MediaPipe
        - joint: The name of the joint (e.g., 'left_elbow')

        Returns:
        - A tuple of (x, y) coordinates of the specified joint
        """
        joint_index_map = {
            'left_shoulder': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': mp.solutions.pose.PoseLandmark.LEFT_HIP,
            'right_hip': mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            'left_knee': mp.solutions.pose.PoseLandmark.LEFT_KNEE,
            'right_knee': mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
        }

        joint_index = joint_index_map[joint]
        landmark = landmarks.landmark[joint_index]

        return landmark.x, landmark.y

    ### Example for defining a function that extracts an angle
    def extract_hip_angle(self, landmarks):
        """
                Extracts the hip angle.

                Parameters:
                - landmarks: The list of pose landmarks from MediaPipe

                Returns:
                - An angle in degrees for the hip
                """
        # extract the x and y coordinates
        left_hip = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
        left_shoulder = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_knee = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
        return self.calculate_angle(left_shoulder, left_hip, left_knee)

    # Extracts the angle of the knee by measuring the angle between the left hip, left knee, and left ankle
    def extract_knee_angle(self, landmarks):
        # extract the x and y coordinates
        left_hip = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y]
        return self.calculate_angle(left_hip, left_knee, left_ankle)


# New hand detection

def detect_hands(self, frame):
    """
    Detect hands and their state (open/closed) in the frame.
    Returns:
    - List of hand information dictionaries with:
        * landmarks: hand landmarks for visualization
        * handedness: identifies left or right hand
        * is_open: boolean indicating if hand is open (for game mechanics)
        * wrist_position: (x, y) coordinates for hand position tracking
    """
    # Convert BGR to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.hands.process(rgb_frame)

    hands_info = []  # Store information for each detected hand

    if results.multi_hand_landmarks and results.multi_handedness:
        # Process each detected hand
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_info = {
                'landmarks': hand_landmarks,  # 21 hand landmarks for visualization
                'handedness': handedness.classification[0].label,  # 'Left' or 'Right'
                'is_open': self._is_hand_open(hand_landmarks),  # Game mechanic: open vs closed hand
                'wrist_position': self._get_wrist_position(hand_landmarks)  # For position tracking
            }
            hands_info.append(hand_info)

    return hands_info


def _is_hand_open(self, hand_landmarks):
    """
    NEW: Determine if hand is open based on finger positions.
    Logic:
    - Calculate distance from wrist to each fingertip
    - Open hand = fingers extended = larger distances from wrist
    - Closed hand = fingers curled = smaller distances from wrist

    Returns: Boolean (True if hand is open, False if closed)
    """
    # Get key landmarks from MediaPipe hand model
    wrist = hand_landmarks.landmark[0]  # Wrist (base of hand)
    thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
    index_tip = hand_landmarks.landmark[8]  # Index finger tip
    middle_tip = hand_landmarks.landmark[12]  # Middle finger tip
    ring_tip = hand_landmarks.landmark[16]  # Ring finger tip
    pinky_tip = hand_landmarks.landmark[20]  # Pinky finger tip

    # Calculate distances from wrist to each fingertip in the calculate_distance_2d
    # Extended fingers create larger distances when hand is open
    # The X_distance is called by the self._calculate_distance (point1, point2) stated in the def _calculate_distance_2d
    thumb_distance = self._calculate_distance_2d(wrist, thumb_tip)
    index_distance = self._calculate_distance_2d(wrist, index_tip)
    middle_distance = self._calculate_distance_2d(wrist, middle_tip)
    ring_distance = self._calculate_distance_2d(wrist, ring_tip)
    pinky_distance = self._calculate_distance_2d(wrist, pinky_tip)

    # Calculate average distance across all fingers
    avg_distance = (thumb_distance + index_distance + middle_distance +
                    ring_distance + pinky_distance) / 5

    # Threshold-based classification
    # Simple but effective for binary open/closed detection
    # Note: This value may need calibration based on testing
    HAND_OPEN_THRESHOLD = 0.15  # Normalized distance threshold

    return avg_distance > HAND_OPEN_THRESHOLD


def _get_wrist_position(self, hand_landmarks):
    """
     Extract wrist position coordinates.
     - the game needs to know where the hand is located on screen
     - Wrist position serves as a stable reference point for hand tracking

     Returns: (x, y) coordinates of wrist (normalized 0-1)
     """
    wrist = hand_landmarks.landmark[0]  # Wrist is landmark 0 in MediaPipe hand model
    return wrist.x, wrist.y


def _calculate_distance_2d(self, point1, point2):
    """
    Calculate 2D distance between two points.
    - Needed for hand state detection (open/closed)
    - Ignores z-coordinate as we're primarily interested in screen position

    Returns: Distance between two points (normalized coordinates)
    """
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

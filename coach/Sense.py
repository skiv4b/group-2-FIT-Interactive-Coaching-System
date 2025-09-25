import cv2
import mediapipe as mp
import math
import numpy as np


# Sense Component: Detect hands using the camera
class Sense:

    def __init__(self):
        # Initialize Mediapipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.17,
            min_tracking_confidence=0.17)

    def detect_hands(self, frame):
        """Detect hands in the frame"""
        return self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def get_hand_data(self, hand_results, hand_type):
        """
        Extract hand landmarks and determine if hand is open or closed
        """
        if not hand_results.multi_hand_landmarks or not hand_results.multi_handedness:
            return None, 'unknown'

        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            handedness = hand_results.multi_handedness[i]
            label = handedness.classification[0].label.lower()

            if label == hand_type:
                # Determine if hand is open or closed based on finger positions
                is_closed = self.is_hand_closed(hand_landmarks.landmark)
                return hand_landmarks.landmark, 'closed' if is_closed else 'open'

        return None, 'unknown'

    def is_hand_closed(self, landmarks):
        """
        Determine if hand is closed based on finger tip positions relative to palm
        """
        if not landmarks:
            return False

        # Get key landmarks
        wrist = landmarks[0]  # Wrist
        thumb_tip = landmarks[4]  # Thumb tip
        index_tip = landmarks[8]  # Index tip
        middle_tip = landmarks[12]  # Middle tip
        ring_tip = landmarks[16]  # Ring tip
        pinky_tip = landmarks[20]  # Pinky tip

        # Calculate distance from finger tips to wrist
        thumb_dist = math.sqrt((thumb_tip.x - wrist.x) ** 2 + (thumb_tip.y - wrist.y) ** 2)
        index_dist = math.sqrt((index_tip.x - wrist.x) ** 2 + (index_tip.y - wrist.y) ** 2)
        middle_dist = math.sqrt((middle_tip.x - wrist.x) ** 2 + (middle_tip.y - wrist.y) ** 2)
        ring_dist = math.sqrt((ring_tip.x - wrist.x) ** 2 + (ring_tip.y - wrist.y) ** 2)
        pinky_dist = math.sqrt((pinky_tip.x - wrist.x) ** 2 + (pinky_tip.y - wrist.y) ** 2)

        # Calculate average distance
        avg_dist = (thumb_dist + index_dist + middle_dist + ring_dist + pinky_dist) / 5

        # If average distance is small, hand is likely closed
        return avg_dist < 0.25

    def get_wrist_position_from_hand(self, hand_landmarks):
        """
        Get wrist position from hand landmarks
        """
        if not hand_landmarks:
            return None

        # Wrist is landmark 0 in hand landmarks
        wrist = hand_landmarks[0]
        return (wrist.x, wrist.y)
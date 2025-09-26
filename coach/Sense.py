import mediapipe as mp
import numpy as np
import math
import cv2

class Sense:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  
            min_detection_confidence=0.5,  
            min_tracking_confidence=0.5
        )

    def detect_hands(self, frame):
        return self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    def get_hand_data(self, hand_results, hand_type):
        
        if not hand_results.multi_hand_landmarks or not hand_results.multi_handedness:
            return None, 'unknown'
        
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            if i < len(hand_results.multi_handedness):
                handedness = hand_results.multi_handedness[i]
                label = handedness.classification[0].label.lower()
                
                if label == hand_type:
                    is_closed = self.is_hand_closed(hand_landmarks.landmark)
                    return hand_landmarks.landmark, 'closed' if is_closed else 'open'
        
        return None, 'unknown'
        
    def is_hand_closed(self, landmarks):
        if not landmarks:
            return False

        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        thumb_dist = math.sqrt((thumb_tip.x - wrist.x) ** 2 + (thumb_tip.y - wrist.y) ** 2)
        index_dist = math.sqrt((index_tip.x - wrist.x) ** 2 + (index_tip.y - wrist.y) ** 2)
        middle_dist = math.sqrt((middle_tip.x - wrist.x) ** 2 + (middle_tip.y - wrist.y) ** 2)
        ring_dist = math.sqrt((ring_tip.x - wrist.x) ** 2 + (ring_tip.y - wrist.y) ** 2)
        pinky_dist = math.sqrt((pinky_tip.x - wrist.x) ** 2 + (pinky_tip.y - wrist.y) ** 2)

        avg_dist = (thumb_dist + index_dist + middle_dist + ring_dist + pinky_dist) / 5

        return avg_dist < 0.25

    def get_wrist_position_from_hand(self, hand_landmarks):
        if not hand_landmarks:
            return None
        
        wrist = hand_landmarks[0]
        return (wrist.x, wrist.y)
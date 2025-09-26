import cv2
import mediapipe as mp
from coach import Sense
from coach import Act
from coach import Think
import numpy as np

def main():
    sense = Sense.Sense()
    act = Act.Act()
    think = Think.Think()
    
    cap = cv2.VideoCapture(0)
    
    print("Starting game...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        results = sense.detect_hands(frame)
        
        # Get hand data
        left_landmarks, left_state = sense.get_hand_data(results, 'left')
        right_landmarks, right_state = sense.get_hand_data(results, 'right')
        
        # Get positions
        left_pos = sense.get_wrist_position_from_hand(left_landmarks)
        right_pos = sense.get_wrist_position_from_hand(right_landmarks)
        
        # Update game
        act.update_game(left_pos, right_pos, left_state, right_state, frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    act.cleanup()

if __name__ == "__main__":
    main()
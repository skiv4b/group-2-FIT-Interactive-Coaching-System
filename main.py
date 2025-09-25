import cv2
import mediapipe as mp
from coach import Sense
from coach import Act
from coach import Think
import numpy as np


def main():
    # Initialize components
    sense = Sense.Sense()
    act = Act.Act()
    think = Think.Think()

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    print("Starting Ball Catching Game with Sound!")
    print("Instructions: Close your hand when a ball approaches to catch it!")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)

            # Sense: Detect hands
            hand_results = sense.detect_hands(frame)

            # Get hand data
            left_hand_landmarks, left_hand_state = sense.get_hand_data(hand_results, 'left')
            right_hand_landmarks, right_hand_state = sense.get_hand_data(hand_results, 'right')

            # Get wrist positions
            left_wrist_pos = sense.get_wrist_position_from_hand(left_hand_landmarks)
            right_wrist_pos = sense.get_wrist_position_from_hand(right_hand_landmarks)

            # Act: Update game visualization
            balls_caught = act.update_game_visualization(left_wrist_pos, right_wrist_pos,
                                                        left_hand_state, right_hand_state, frame)

            # Display mediapipe view
            act.display_mediapipe_view(frame, hand_results)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Game interrupted by user")
    except Exception as e:
        print(f"Error during game execution: {e}")
    finally:
        # Cleanup
        cap.release()
        act.cleanup()
        cv2.destroyAllWindows()
        print("Game closed successfully!")

if __name__ == "__main__":
    main()
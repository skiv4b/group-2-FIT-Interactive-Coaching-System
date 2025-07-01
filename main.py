import cv2
import mediapipe as mp
from coach import Sense
from coach import Think
from coach import Act

import numpy as np


# Main Program Loop
def main():
    """
    Main function to initialize the exercise tracking application.

    This function sets up the webcam feed, initializes the Sense, Think, and Act components,
    and starts the main loop to continuously process frames from the webcam.
    """

    # Initialize the components: Sense for input, Think for decision-making, Act for output
    sense = Sense.Sense()
    act = Act.Act()
    think = Think.Think(act)

    # Initialize the webcam capture
    cap = cv2.VideoCapture(0)  # Use the default camera (0)

    # Main loop to process video frames
    while cap.isOpened():

        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Sense: Detect joints
        joints = sense.detect_joints(frame)
        landmarks = joints.pose_landmarks

        # If landmarks are detected, calculate the elbow angle
        if landmarks:
            # Extract joint coordinates for the left arm
            # For this example, we will use specific landmark indexes for shoulder, elbow, and wrist
            shoulder = sense.extract_joint_coordinates(landmarks, 'left_shoulder')
            elbow = sense.extract_joint_coordinates(landmarks, 'left_elbow')
            wrist = sense.extract_joint_coordinates(landmarks, 'left_wrist')

            # Calculate the elbow angle
            elbow_angle_mvg = sense.calculate_angle(shoulder, elbow, wrist)

            # Think: Next, give the angles to the decision-making component and make decisions based on joint data
            think.update_state(elbow_angle_mvg, sense.previous_angle)

            # We'll save the previous angle for later comparison
            sense.previous_angle = elbow_angle_mvg

            decision = think.state

            # Act: Provide feedback to the user.
            act.provide_feedback(decision, frame=frame, joints=joints, elbow_angle_mvg=elbow_angle_mvg)
            # Render the balloon visualization
            act.visualize_balloon()

            # think.check_for_timeout()

        # Exit if the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

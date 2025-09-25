# Act Component: Provide feedback to the user

import mediapipe as mp
import cv2
import numpy as np
import random
import pygame

# Act Component: Visualization to motivate user, visualization such as the skeleton and debugging information.
# Things to add: Other graphical visualization, a proper GUI, more verbal feedback
class Act:

    def __init__(self, width = 500, height=500):
        self.width = width
        self.height = height
        self.left_grip = False
        self.right_grip = True

        # Initialize pygame mixer (fast sound playback)
        pygame.mixer.init()

        # Preload sounds into memory
        self.game_start = pygame.mixer.Sound("files/Game Start.mp3")
        self.game_over = pygame.mixer.Sound("files/Game Over.mp3")
        self.capture = pygame.mixer.Sound("files/Capturing the Ball.mp3")

        # Track capture channel
        self.capture_channel = None

    def left_grip(self):
        self.left_grip = True

    def right_grip (self):
        self.right_grip = True
    
    def play_game_start(self):
        self.game_start.play()

    def play_game_over(self):
        self.game_over.play()

    def play_capture(self):
        self.capture.play()

    def play_capture(self):
        # Play on its own channel so we can track it
        self.capture_channel = self.capture.play()

    def visualize_balloon(self):
        radius = 50
        y_pos = -radius
        x_pos = random.randint(radius, self.width - radius)
        captured = False
        self.score = 0   # 🔹 initialize score

        # Load images
        left = cv2.imread("files/LeftHandOpen.png", cv2.IMREAD_UNCHANGED)
        right = cv2.imread("files/RightHandOpen.png", cv2.IMREAD_UNCHANGED)
        left_closed = cv2.imread("files/LeftHandClosed.png", cv2.IMREAD_UNCHANGED)
        right_closed = cv2.imread("files/RightHandClosed.png", cv2.IMREAD_UNCHANGED)

        if left is None or right is None or left_closed is None or right_closed is None:
            print("Error: Could not load one or multiple images.")
            return

        # Resize
        left = cv2.resize(left, (200, 200))
        right = cv2.resize(right, (200, 200))
        left_closed = cv2.resize(left_closed, (150, 150))
        right_closed = cv2.resize(right_closed, (150, 150))

        while True:
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Draw balloon only if not captured
            if not captured:
                cv2.circle(img, (x_pos, y_pos), radius, (0, 0, 255), -1)

            # Hand overlay positions
            if not self.left_grip:
                lx, ly = 10, 300
                self.overlay_with_alpha(img, left, lx, ly)
            else:
                lx, ly = 20, 350
                self.overlay_with_alpha(img, left_closed, lx, ly)

            if not self.right_grip:
                rx, ry = 300, 300
                self.overlay_with_alpha(img, right, rx, ry)
            else:
                rx, ry = 330, 350
                self.overlay_with_alpha(img, right_closed, rx, ry)

            # ---- Collision detection ----
            def check_hand_collision(cx, cy, r, x, y, w, h):
                within_x = (x <= cx <= x + w)
                hand_mid_y = y + h // 2
                within_y = abs(cy - hand_mid_y) <= r
                return within_x and within_y

            if not captured:
                if self.left_grip:
                    h, w = left_closed.shape[:2]
                    if check_hand_collision(x_pos, y_pos, radius, lx, ly, w, h):
                        self.play_capture()
                        self.score += 1   # 🔹 increase score
                        captured = True

                if self.right_grip:
                    h, w = right_closed.shape[:2]
                    if check_hand_collision(x_pos, y_pos, radius, rx, ry, w, h):
                        self.play_capture()
                        self.score += 1   # 🔹 increase score
                        captured = True

            # ---- Respawn only when sound finished ----
            if captured:
                if self.capture_channel is None or not self.capture_channel.get_busy():
                    y_pos = -radius
                    x_pos = random.randint(radius, self.width - radius)
                    captured = False

            # ---- Draw score ----
            cv2.putText(img, f"Score: {self.score}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Show frame
            cv2.imshow("Boulder climb!", img)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

            # Move balloon only if not captured
            if not captured:
                y_pos += 5
                if y_pos - radius > self.height:
                    y_pos = -radius
                    x_pos = random.randint(radius, self.width - radius)


        # Wait for 1 ms and check if the window should be closed
        cv2.waitKey(1)

    def overlay_with_alpha(self, background, overlay, x, y):
        """
        Draw overlay RGBA image onto background BGR image at (x,y).
        """
        h, w = overlay.shape[:2]

        # Clip if overlay doesn't fit
        if x + w > background.shape[1] or y + h > background.shape[0]:
            return

        if overlay.shape[2] == 4:  # RGBA image
            bgr = overlay[:, :, :3]
            alpha = overlay[:, :, 3] / 255.0

            roi = background[y:y+h, x:x+w]

            # Blend each channel
            for c in range(3):
                roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * bgr[:, :, c]

            background[y:y+h, x:x+w] = roi
        else:  # No alpha channel
            background[y:y+h, x:x+w] = overlay

    def provide_feedback(self, decision, frame, joints, elbow_angle_mvg):
        """
        Displays the skeleton and some text using open cve.

        :param decision: The decision in which state the user is from the think component.
        :param frame: The currently processed frame form the webcam.
        :param joints: The joints extracted from mediapipe from the current frame.
        :param elbow_angle_mvg: The moving average from the left elbow angle.

        """

        mp.solutions.drawing_utils.draw_landmarks(frame, joints.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # Define the number and text to display
        number = elbow_angle_mvg
        text = " "
        if decision == 'flexion':
            text = "You are flexing your elbow! %s" % number
        elif decision == 'extension':
            text = "You are extending your elbow! %s" % number


        # Set the position, font, size, color, and thickness for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .9
        font_color = (0, 0, 0)  # White color in BGR
        thickness = 2

        # Define the position for the number and text
        text_position = (50, 50)

        # Draw the text on the image
        cv2.putText(frame, text, text_position, font, font_scale, font_color, thickness)

        # Display the frame (for debugging purposes)
        cv2.imshow('Sport Coaching Program', frame)

        cv2.waitKey(1)

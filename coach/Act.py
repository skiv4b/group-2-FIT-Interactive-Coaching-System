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
        self.right_grip = False

        # Initialize pygame mixer (fast sound playback)
        pygame.mixer.init()

        # Preload sounds into memory
        self.game_start = pygame.mixer.Sound("files/Game Start.mp3")
        self.game_over = pygame.mixer.Sound("files/Game Over.mp3")
        self.capture = pygame.mixer.Sound("files/Capturing the Ball.mp3")

        # Track capture channel
        self.capture_channel = None

        self.balloon_radius = 50
        self.balloon_y = -self.balloon_radius
        self.balloon_x = random.randint(self.balloon_radius, self.width - self.balloon_radius)
        self.balloon_captured = False
        self.score = 0

    def left_grip_state(self):
        self.left_grip = True

    def right_grip_state(self):
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
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw balloon only if not captured
        if not self.balloon_captured:
            cv2.circle(img, (self.balloon_x, self.balloon_y), self.balloon_radius, (0, 0, 255), -1)

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

        # Hand overlay positions
        if not self.left_grip:
            lx, ly = 50, 300
            self.overlay_with_alpha(img, left, lx, ly)
        else:
            lx, ly = 60, 350
            self.overlay_with_alpha(img, left_closed, lx, ly)

        if not self.right_grip:
            rx, ry = 250, 300
            self.overlay_with_alpha(img, right, rx, ry)
        else:
            rx, ry = 280, 350
            self.overlay_with_alpha(img, right_closed, rx, ry)

        # ---- Collision detection function ----
        def check_hand_collision(cx, cy, r, x, y, w, h):
            """
            cx, cy: balloon center
            r: balloon radius
            x, y: top-left of hand
            w, h: width and height of hand image
            """
            # Full width including balloon edges
            within_x = (x - r <= cx <= x + w + r)
            
            # Vertical collision around hand middle
            hand_mid_y = y + h // 2
            within_y = (hand_mid_y - r <= cy <= hand_mid_y + r)
            
            return within_x and within_y

        # ---- Balloon movement ----
        if not self.balloon_captured:
            self.balloon_y += 10
            if self.balloon_y - self.balloon_radius > self.height:
                self.balloon_y = -self.balloon_radius
                self.balloon_x = random.randint(self.balloon_radius, self.width - self.balloon_radius)

            # Closed hand positions for collision
            hands = []
            if self.left_grip:
                hands.append((lx, ly, left_closed.shape[1], left_closed.shape[0]))
            if self.right_grip:
                hands.append((rx, ry, right_closed.shape[1], right_closed.shape[0]))

            # Check collisions with closed hands
            for (hx, hy, hw, hh) in hands:
                if check_hand_collision(self.balloon_x, self.balloon_y, self.balloon_radius, hx, hy, hw, hh):
                    self.score += 1

                    # Play capture sound safely
                    if getattr(self, 'capture_channel', None) is not None:
                        self.capture_channel.stop()
                    self.capture_channel = self.capture.play()

                    self.balloon_captured = True
                    break

        # ---- Respawn balloon after capture sound finishes ----
        if self.balloon_captured:
            if getattr(self, 'capture_channel', None) is None or not self.capture_channel.get_busy():
                self.balloon_captured = False
                self.balloon_y = -self.balloon_radius
                self.balloon_x = random.randint(self.balloon_radius, self.width - self.balloon_radius)

        # Draw score
        cv2.putText(img, f"Score: {self.score}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("Boulder climb!", img)
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

        text = " "
        if decision == 'flexion':
            text = "You are climbing!" 
        elif decision == 'extension':
            text = "Your hand is open!"


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

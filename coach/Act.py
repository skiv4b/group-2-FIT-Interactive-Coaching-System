import cv2
import numpy as np
import random
import time
import pygame

class Act:  
    def __init__(self):
        self.game_window_size = (800, 600)
        self.ball_radius = 30
        self.balls = []  
        self.score = 0
        self.last_ball_time = time.time()
        self.ball_interval = 2.0  
        
        self.load_hand_images()
        self.setup_sound()
        
        self.last_catch_time = 0
        self.catch_sound_cooldown = 0.3  

    def load_hand_images(self):
        try:
            self.left_hand_open = cv2.imread("files/LeftHandOpen.png", cv2.IMREAD_UNCHANGED)
            self.left_hand_closed = cv2.imread("files/LeftHandClosed.png", cv2.IMREAD_UNCHANGED)
            self.right_hand_open = cv2.imread("files/RightHandOpen.png", cv2.IMREAD_UNCHANGED)
            self.right_hand_closed = cv2.imread("files/RightHandClosed.png", cv2.IMREAD_UNCHANGED)
            
            if (self.left_hand_open is None or self.left_hand_closed is None or 
                self.right_hand_open is None or self.right_hand_closed is None):
                raise Exception("Could not load one or more hand images")
            
            self.left_hand_open = cv2.resize(self.left_hand_open, (200, 200))
            self.right_hand_open = cv2.resize(self.right_hand_open, (200, 200))
            self.left_hand_closed = cv2.resize(self.left_hand_closed, (150, 150))
            self.right_hand_closed = cv2.resize(self.right_hand_closed, (150, 150))
            
        except Exception as e:
            print(f"Could not load hand images: {e}")
            self.left_hand_open = None
            self.left_hand_closed = None
            self.right_hand_open = None
            self.right_hand_closed = None

    def setup_sound(self):
        try:
            pygame.mixer.init()
            
            # Probeer haar geluidsbestanden te laden
            try:
                self.capture_sound = pygame.mixer.Sound("files/Capturing the Ball.mp3")
                self.use_file_sounds = True
            except:
                self.use_file_sounds = False
                self.capture_sound = self.create_beep_sound(800, 0.2)
                self.miss_sound = self.create_beep_sound(300, 0.3)
                self.new_ball_sound = self.create_beep_sound(600, 0.1)
            
        except Exception as e:
            print(f"Sound initialization failed: {e}")
            self.use_file_sounds = False


    def play_sound(self, sound):
        if sound and pygame.mixer.get_init():
            try:
                sound.play()
            except:
                pass

    def overlay_with_alpha(self, background, overlay, x, y):
        if overlay is None:
            return
               
        y = max(0, min(y, background.shape[0] - overlay.shape[0]))
        x = max(0, min(x, background.shape[1] - overlay.shape[1]))
        
        overlay_height, overlay_width = overlay.shape[:2]
        background_height, background_width = background.shape[:2]
        
        start_y = max(0, y)
        end_y = min(background_height, y + overlay_height)
        start_x = max(0, x)
        end_x = min(background_width, x + overlay_width)
        
        overlay_start_y = max(0, -y)
        overlay_end_y = overlay_start_y + (end_y - start_y)
        overlay_start_x = max(0, -x)
        overlay_end_x = overlay_start_x + (end_x - start_x)
        
        if overlay.shape[2] == 4:
            overlay_rgb = overlay[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x, :3]
            overlay_alpha = overlay[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha
            
            background[start_y:end_y, start_x:end_x] = (
                background[start_y:end_y, start_x:end_x] * background_alpha + 
                overlay_rgb * overlay_alpha
            )
        else:
            background[start_y:end_y, start_x:end_x] = overlay[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x, :3]

    def update_game_visualization(self, left_wrist_pos, right_wrist_pos,
                                  left_hand_state, right_hand_state, frame):
        current_time = time.time()
        
        if current_time - self.last_ball_time > self.ball_interval:
            self.add_new_ball()
            self.last_ball_time = current_time
            if not self.use_file_sounds:
                self.play_sound(self.new_ball_sound)
        
        game_frame = np.ones((self.game_window_size[1], self.game_window_size[0], 3), dtype=np.uint8) * 200
        
        balls_caught_this_frame = self.update_balls(left_wrist_pos, right_wrist_pos, left_hand_state, right_hand_state)
        
        if balls_caught_this_frame > 0 and current_time - self.last_catch_time > self.catch_sound_cooldown:
            self.play_sound(self.capture_sound)
            self.last_catch_time = current_time
        
        self.draw_balls(game_frame)
        
        self.draw_hands_with_png(game_frame, left_wrist_pos, right_wrist_pos, left_hand_state, right_hand_state)
        
        self.draw_ui(game_frame)
        
        cv2.imshow('Ball Catching Game', game_frame)
        
        return balls_caught_this_frame

    def add_new_ball(self):
        x = random.randint(self.ball_radius, self.game_window_size[0] - self.ball_radius)
        speed = random.uniform(2.0, 5.0)  
        self.balls.append([x, 0, speed, False, False])  

    def update_balls(self, left_wrist_pos, right_wrist_pos, left_hand_state, right_hand_state):
        hands = []
        balls_caught = 0
    
        if left_hand_state == 'closed' and left_wrist_pos:
            x = int(left_wrist_pos[0] * self.game_window_size[0])
            y = int(left_wrist_pos[1] * self.game_window_size[1])
            hands.append((x, y))
        
        if right_hand_state == 'closed' and right_wrist_pos:
            x = int(right_wrist_pos[0] * self.game_window_size[0])
            y = int(right_wrist_pos[1] * self.game_window_size[1])
            hands.append((x, y))
        
        for ball in self.balls[:]:
            ball[1] += ball[2]  
            
            if not ball[3]:  
                for hand_x, hand_y in hands:
                    distance = np.sqrt((ball[0] - hand_x)**2 + (ball[1] - hand_y)**2)
                    if distance < self.ball_radius + 20: 
                        ball[3] = True  
                        ball[4] = False  
                        self.score += 1
                        balls_caught += 1
                        break
            
            if ball[1] > self.game_window_size[1] + 50:
                if not ball[3] and not self.use_file_sounds:  
                    self.play_sound(self.miss_sound)
                self.balls.remove(ball)
        
        return balls_caught

    def draw_balls(self, frame):
        """Draw all balls on the frame - onze simpele ballen"""
        for ball in self.balls:
            x, y, speed, caught, sound_played = ball
            color = (0, 255, 0) if caught else (0, 255, 255)  
            
            cv2.circle(frame, (int(x), int(y)), self.ball_radius, color, -1)
            cv2.circle(frame, (int(x), int(y)), self.ball_radius, (255, 255, 255), 2)
            
            if caught:
                cv2.circle(frame, (int(x), int(y)), self.ball_radius + 5, (0, 200, 0), 3)

    def draw_hands_with_png(self, frame, left_wrist_pos, right_wrist_pos, left_hand_state, right_hand_state):
        """Draw hand positions with PNG images"""
        if left_wrist_pos:
            x = int(left_wrist_pos[0] * self.game_window_size[0])
            y = int(left_wrist_pos[1] * self.game_window_size[1])
            
            if self.left_hand_open is not None and self.left_hand_closed is not None:
                if left_hand_state == 'closed':
                    self.overlay_with_alpha(frame, self.left_hand_closed, x - 75, y - 75)
                else:
                    self.overlay_with_alpha(frame, self.left_hand_open, x - 100, y - 100)
            else:
                color = (0, 0, 255) if left_hand_state == 'closed' else (255, 0, 0)
                size = 25 if left_hand_state == 'closed' else 20
                cv2.circle(frame, (x, y), size, color, -1)
                cv2.circle(frame, (x, y), size, (255, 255, 255), 2)
                cv2.putText(frame, "Left", (x-20, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if right_wrist_pos:
            x = int(right_wrist_pos[0] * self.game_window_size[0])
            y = int(right_wrist_pos[1] * self.game_window_size[1])
            
            if self.right_hand_open is not None and self.right_hand_closed is not None:
                if right_hand_state == 'closed':
                    self.overlay_with_alpha(frame, self.right_hand_closed, x - 75, y - 75)
                else:
                    self.overlay_with_alpha(frame, self.right_hand_open, x - 100, y - 100)
            else:
                color = (0, 0, 255) if right_hand_state == 'closed' else (255, 0, 0)
                size = 25 if right_hand_state == 'closed' else 20
                cv2.circle(frame, (x, y), size, color, -1)
                cv2.circle(frame, (x, y), size, (255, 255, 255), 2)
                cv2.putText(frame, "Right", (x-20, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_ui(self, frame):
        cv2.rectangle(frame, (0, 0), (800, 130), (200, 200, 200), -1)
        cv2.rectangle(frame, (0, 530), (800, 600), (200, 200, 200), -1)
        
        # Title and score
        cv2.putText(frame, "BALL CATCHING GAME", (250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        cv2.putText(frame, f"Score: {self.score}", (50, 550),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Instructions
        cv2.putText(frame, "Close your hand to catch balls!", (250, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Ball counter
        cv2.putText(frame, f"Balls on screen: {len(self.balls)}", (500, 550),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def display_mediapipe_view(self, frame, hand_results):
        """Display the mediapipe hand detection view"""
        import mediapipe as mp
        
        display_frame = frame.copy()
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        
        cv2.putText(display_frame, "Hand Detection View - Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Hand Detection View', display_frame)

    def cleanup(self):
        """Clean up resources"""
        if pygame.mixer.get_init():
            pygame.mixer.quit()
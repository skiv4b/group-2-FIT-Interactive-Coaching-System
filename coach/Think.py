class Think:
    def __init__(self):
        self.score = 0
        self.balls_caught = 0
        self.balls_missed = 0

    def get_difficulty_level(self):
        total_balls = self.balls_caught + self.balls_missed
        if total_balls == 0:
            return 1.0
        
        success_rate = self.balls_caught / total_balls
        return min(2.0, max(0.5, success_rate))  
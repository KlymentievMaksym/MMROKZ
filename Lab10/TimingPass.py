import time
import numpy as np
from scipy.stats import ttest_ind
from pynput import keyboard
import sys

class TimingPass:
    def __init__(self, target_phrase="mama"):
        self.target_phrase = target_phrase
        self.timings = []
        self.last_press = 0

    def on_press(self, key):
        current_time = time.time()
        
        if self.last_press == 0:
            self.last_press = current_time
            return

        if key == keyboard.Key.backspace:
            if self.timings:
                self.timings.pop()
            
            self.last_press = current_time
            return

        if key == keyboard.Key.enter:
            return False

        delay = current_time - self.last_press
        self.timings.append(delay)
        self.last_press = current_time

    def capture_attempt(self, attempt_name):
        self.timings = []
        self.last_press = 0

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        text = input(f"[{attempt_name}] Enter '{self.target_phrase}' and then press Enter:\n>>> ")
        listener.stop()
            
        print(f"[Length] {len(self.timings)}")
        return np.array(self.timings)

    def train(self, iterations: int = 5):
        print("[Training]")
        clean_profiles = []
        
        for i in range(iterations):
            data = self.capture_attempt(f"{i+1}/{iterations}")

            # Фільтрація грубих помилок (якщо довжина не збігається)
            # В реальності тут має бути алгоритм з методички (відкидання викидів)
            expected_intervals = len(self.target_phrase) - 1
            if len(data) != expected_intervals:
                print(f"[Error] Expected: {expected_intervals}, Received: {len(data)}")
            else:
                clean_profiles.append(data)
        
        if not clean_profiles:
            print("[Error] No data collected")
            sys.exit()

        # Об'єднуємо всі спроби в один масив для кожного інтервалу
        # Для спрощення беремо середнє затримки по всіх спробах
        self.reference_data = np.concatenate(clean_profiles)
        
        mean_val = np.mean(self.reference_data)
        var_val = np.var(self.reference_data)
        print(f"[Mean] {mean_val:.4f} | [Variance] {var_val:.4f}")

    def test(self, iterations: int = 5):
        print(f"[Testing]")
        for iteration in range(iterations):
            sample = self.capture_attempt("Test")
            
            if len(sample) == 0:
                print("Empty sample")
                return

            t_stat, p_value = ttest_ind(self.reference_data, sample, equal_var=False)
            
            print("-" * 45)
            print(f"T: {t_stat:.4f}")
            print(f"P-value:   {p_value:.4f}")
            print("-" * 45)

            alpha = 0.05

            res = p_value > alpha
            if res:
                print("[Access Granted]")
            else:
                print("[Access Denied]")
            

if __name__ == "__main__":
    app = TimingPass("dlagnytor") 
    app.train(iterations=5)
    app.test(iterations=3)
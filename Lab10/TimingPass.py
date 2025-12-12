import time
import numpy as np
from scipy.stats import ttest_ind, f
from pynput import keyboard
import sys
import os

class TimingPass:
    def __init__(self, target_phrase="dlagnytor"):
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
        text = input(f"[{attempt_name}] Enter '{self.target_phrase}':\n>>> ")
        listener.stop()
        if text != self.target_phrase:
            print("[Access Denied] Wrong phrase")
            return self.capture_attempt(attempt_name)
        return np.array(self.timings)

    def _load_content(self, file_path: str) -> np.array:
        with open(file_path, "r") as f:
            return np.genfromtxt(f, delimiter=",", dtype=float)

    def filter_outliers(self, data):
        filtered_data = data.copy()
        if len(data) == 0: return data
        mean = np.mean(data)
        std = np.std(data)

        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        for row in data:
            if (row < lower_bound).any() or (row > upper_bound).any():
                filtered_data = np.delete(filtered_data, np.where(filtered_data == row), axis=0)
        # print(filtered_data)

        return filtered_data

    def train(self, iterations: int = 5, training_data_path: str = None, save: bool = False, read: bool = False, rewrite: bool = False):
        print("[Training]")
        if read and os.path.exists(training_data_path) and os.path.isfile(training_data_path) and len(self._load_content(training_data_path)) > 0:
            self.reference_data = self._load_content(training_data_path)
        else:
            clean_profiles = []
            
            for i in range(iterations):
                data = self.capture_attempt(f"{i+1}/{iterations}")
                clean_profiles.append(data)

            if not clean_profiles:
                print("[Error] No data collected")
                sys.exit()

            self.reference_data = np.array(clean_profiles)

        # print(self.reference_data.shape)
        self.reference_data = self.filter_outliers(self.reference_data)
        # print(self.reference_data.shape)

        if save:
            if not rewrite:
                self.reference_data = np.concatenate((self.reference_data, self.filter_outliers(self._load_content(training_data_path))))

            np.savetxt(training_data_path, self.reference_data, delimiter=",")

    def fisher_test(self, data1, data2):
        var1 = np.var(data1, ddof=1)
        var2 = np.var(data2, ddof=1)
        
        # F = S_max^2 / S_min^2
        if var1 > var2:
            f_stat = var1 / var2
            df1, df2 = len(data1) - 1, len(data2) - 1
        else:
            f_stat = var2 / var1
            df1, df2 = len(data2) - 1, len(data1) - 1
            
        p_value = 1 - f.cdf(f_stat, df1, df2)
        return p_value * 2

    def test(self, alpha: float = 0.05):
        print(f"[Testing]")
        samples = self.capture_attempt("Test")
        if len(samples) == 0:
            print("[Error] No data")
            return
        samples = np.array(samples)

        ref_flat = self.reference_data.flatten()
        p_value_f = self.fisher_test(ref_flat, samples)
        variances_equal = p_value_f > alpha

        t_stat, p_value = ttest_ind(ref_flat, samples, equal_var=variances_equal)

        print("-" * 45)
        print(f"[Fisher P-value] {p_value_f:.4f} {"EQUAL" if variances_equal else "NOT EQUAL"}")
        print(f"[T-test P-value] {p_value:.4f}")
        print("-" * 45)

        res = p_value > alpha
        if res:
            print("[Access Granted]")
        else:
            print("[Access Denied]")
        return res

    def calculate_errors(self, tests_amount: int = 100, alpha: float = 0.05):
        print(f"[Calculating errors]")
        
        if len(self.reference_data) == 0:
            print("[Error] Спочатку пройдіть навчання!")
            return

        ref_mean = np.mean(self.reference_data)
        ref_std = np.std(self.reference_data)

        imposter_data = self.reference_data * 2
        imp_mean = np.mean(imposter_data)
        imp_std = np.std(imposter_data)

        sample_size = self.reference_data.shape[1]
        false_rejections = 0
        for _ in range(tests_amount):
            simulated_attempt = np.random.normal(ref_mean, ref_std, sample_size)
            t_stat, p_val = ttest_ind(self.reference_data.flatten(), simulated_attempt, equal_var=False)
            if p_val <= alpha:
                false_rejections += 1

        p1 = false_rejections / tests_amount
        print(f"[Score] P1: {p1:.2f} ({p1*100:.2f}%)")

        false_acceptances = 0
        for _ in range(tests_amount):
            simulated_impostor = np.random.normal(imp_mean, imp_std, sample_size)
            t_stat, p_val = ttest_ind(self.reference_data.flatten(), simulated_impostor, equal_var=False)
            if p_val > alpha:
                false_acceptances += 1

        p2 = false_acceptances / tests_amount
        print(f"[Score] P2: {p2:.2f} ({p2*100:.2f}%)")


if __name__ == "__main__":
    app = TimingPass("dlagnytor")
    # app = TimingPass("mama")
    app.train(iterations=20, training_data_path="./Lab10/Data/training_data.txt", save=False, read=True, rewrite=True)
    # while True:
    app.test(alpha=.1)
    app.calculate_errors(tests_amount=1000, alpha=.1)
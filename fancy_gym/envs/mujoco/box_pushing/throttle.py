import time
import matplotlib.pyplot as plt
from collections import deque


class Throttle:
    """Utility class to make a loop run at a specific rate, useful to run real-life rollouts at a consistent rate"""

    def __init__(self, target_hz: float, busy_wait: bool = False):
        self.busy_wait = busy_wait
        self.target_hz = target_hz
        self.start_time = time.perf_counter()
        self.history = deque(maxlen=100)
        self.count = 0

    def tick(self):
        self.count += 1
        target_time = self.count / self.target_hz + self.start_time

        error = target_time - time.perf_counter()

        if error > 0:
            self._sleep(error)
        else:
            print("Warning, Throttle got called to slowly")

        self.history.append(target_time - time.perf_counter())

    def _sleep(self, seconds: float):
        if not self.busy_wait:
            time.sleep(seconds)
            return

        target = time.perf_counter() + seconds
        while time.perf_counter() < target:
            pass

import torch
import random
import os
import numpy as np
import collections
import time
import threading


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class QLock:
    """
    A QueueLock implementation, see: https://stackoverflow.com/questions/19688550/how-do-i-queue-my-python-locks
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.waiters = collections.deque()
        self.count = 0

    def acquire(self):
        self.lock.acquire()
        if self.count:
            new_lock = threading.Lock()
            new_lock.acquire()
            self.waiters.append(new_lock)
            self.lock.release()
            new_lock.acquire()
            self.lock.acquire()
        self.count += 1
        self.lock.release()

    def release(self):
        with self.lock:
            if not self.count:
                raise ValueError("lock not acquired")
            self.count -= 1
            if self.waiters:
                self.waiters.popleft().release()
        time.sleep(0.01)

    def locked(self):
        return self.count > 0
    
    def __enter__(self):
        self.acquire()
    
    def __exit__(self, type, val, traceback):
        self.release()

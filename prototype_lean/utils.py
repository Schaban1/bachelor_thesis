import collections
import json
import logging
import numpy as np
import os
import queue
import random
import threading
import threading
import time
import torch
from concurrent.futures import Future
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from typing import Callable


def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ProducerConsumer:
    def __init__(self, numworkers: int = 1) -> None:
        self.queue = Queue()
        self.running = True
        self.worker_threads = []
        for _ in range(numworkers):
            t = threading.Thread(target=self.__worker_main, daemon=True)
            t.start()
            self.worker_threads.append(t)

    def __worker_main(self) -> None:
        while self.running:
            future, task = self.queue.get()
            if future is None:
                break
            try:
                result = task()
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    def do_work(self, work: Callable) -> Future:
        future = Future()
        self.queue.put((future, work))
        return future

    def destroy(self) -> None:
        pass

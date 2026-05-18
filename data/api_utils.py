# -*- coding: utf-8 -*-
"""Shared API helpers for Tushare and AkShare calls."""

import os
import random
import threading
import time


class SafeAPICaller:
    def __init__(
        self,
        min_interval=1.5,
        max_retries=3,
        retry_base_delay=4.0,
        jitter=(0.2, 0.5),
        data_source="api",
    ):
        self.min_interval = float(min_interval)
        self.max_retries = int(max_retries)
        self.retry_base_delay = float(retry_base_delay)
        self.jitter = jitter
        self.data_source = data_source
        self._lock = threading.Lock()
        self._last_request_time = 0.0

    def _reserve_slot(self):
        with self._lock:
            now = time.time()
            next_time = max(now, self._last_request_time + self.min_interval)
            self._last_request_time = next_time
        wait = next_time - now
        if wait > 0:
            time.sleep(wait)
        if self.jitter:
            low, high = self.jitter
            if high and high > 0:
                time.sleep(random.uniform(low, high))

    def __call__(self, func, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                self._reserve_slot()
                return func(*args, **kwargs)
            except Exception as exc:
                wait = self.retry_base_delay * (attempt + 1)
                print(f"  {self.data_source} 调用失败 ({attempt + 1}/{self.max_retries}): {exc}, 等待 {wait}s")
                time.sleep(wait)
        return None


def resolve_tushare_token(token=None, context=None):
    resolved = token or os.getenv("TUSHARE_TOKEN")
    if not resolved:
        suffix = f"；{context}" if context else ""
        raise ValueError(f"缺少 TUSHARE_TOKEN，请设置环境变量或通过 --token 传入{suffix}")
    return resolved

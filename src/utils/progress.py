"""Simple progress bar for CLI."""

import sys
import time

class SimpleProgressBar:
    def __init__(self, total, prefix='Progress', width=40):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0
        self.start_time = time.time()
    
    def update(self, current=None, suffix=''):
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = ""
        
        sys.stdout.write(f'\r{self.prefix}: |{bar}| {self.current}/{self.total} {eta_str} {suffix}')
        sys.stdout.flush()
        
        if self.current >= self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()
    
    def finish(self):
        self.update(self.total)

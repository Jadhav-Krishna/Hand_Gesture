import time
from datetime import datetime
from collections import deque

class GestureHistory:
    def __init__(self, max_size=10):
        self.history = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add_item(self, gesture):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history.appendleft((gesture, timestamp))
    
    def get_history(self):
        return list(self.history)
    
    def clear(self):
        self.history.clear()
    
    def get_most_frequent(self, time_window=None):
        if not self.history:
            return None
        
        if time_window:
            current_time = time.time()
            filtered_history = [item for item, timestamp in self.history 
                               if (current_time - timestamp) <= time_window]
        else:
            filtered_history = [item for item, _ in self.history]
        
        if not filtered_history:
            return None
        
        # Count occurrences
        counts = {}
        for item in filtered_history:
            if item in counts:
                counts[item] += 1
            else:
                counts[item] = 1
        
        # Find the most frequent
        most_frequent = max(counts.items(), key=lambda x: x[1])
        return most_frequent[0]
import time    
import os


class ProgressBar:
    def __init__(self, num_iters, header=None) -> None:
        self.t0 = time.time()
        self.n = num_iters
        self.term_cols = 20
        #     self.term_cols = (os.get_terminal_size().columns)*0.25
        
        if header is not None:
            print(header)
        print(f"[{'.'*int(self.term_cols/self.n)}] loop started..!", end="\r")
        
    def update(self, i: int):
        
        tc_points = int(((i+1)/self.n) * self.term_cols)
        avg_time = (time.time()-self.t0)/(i+1)
        percent = ((i+1)/self.n)*100
        #
        completion = f"{percent :04.2f}%"
        time_elapsed = f" | Elapsed time: {avg_time*(i+1):04.1f} seconds"
        time_remaining = f" | Remaining time: {(avg_time*(self.n-(i+1))): 04.1f} seconds"
        _bar = f"[{'=' * tc_points + '>' + '.' * int(self.term_cols - tc_points)}]"
        print(f"{_bar}{completion} {time_elapsed} {time_remaining}", end="\r")
        if i==self.n-1:
            print("\n")
        return
  
    
if __name__== "__main__":
    N = 325
    bar = ProgressBar(N, header="Testing1")   
    for k in range(N):
        # stuff goes here #
        time.sleep(0.0001)
        # stuff goes here #
        bar.update(k)
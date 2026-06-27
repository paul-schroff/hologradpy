import time
import torch


# TODO: Move to a more sensible location.
class Timer:
    def __init__(self, use_cuda: bool = False, verbose: bool = False) -> None:
        """Timer class to measure elapsed time for CUDA and non-CUDA
        operations."""
        self.elapsed_time: float = None
        self.use_cuda: bool = use_cuda
        self.verbose: bool = verbose

        if use_cuda:
            self.start_event = None
            self.stop_event = None
        else:
            self.start_time: float = None
            self.stop_time: float = None

    def start(self):
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.stop_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()

        self.start_time = time.time()

        if self.verbose:
            date = time.strftime("%d-%m-%y__%H-%M-%S", time.localtime())
            print("Calculation start: %s\n" % date)

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")

        self.stop_time = time.time()

        if self.use_cuda:
            self.stop_event.record()
            torch.cuda.synchronize()
            self.elapsed_time = self.start_event.elapsed_time(self.stop_event) / 1e3
        else:
            self.elapsed_time = self.stop_time - self.start_time

        if self.verbose:
            print(
                f"Ran for {(self.elapsed_time // 60):.0f} minutes and "
                + f"{(self.elapsed_time % 60):.2f} seconds."
            )

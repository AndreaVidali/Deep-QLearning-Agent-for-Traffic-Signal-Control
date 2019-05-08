import random

# HANDLES THE MEMORY
class Memory:
    def __init__(self, memory_size):
        self._memory_size = memory_size
        self._samples = []

    # ADD A SAMPLE INTO THE MEMORY
    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._memory_size:
            self._samples.pop(0)  # if the length is greater than the size of memory, remove the oldest element

    # GET n_samples SAMPLES RANDOMLY FROM THE MEMORY
    def get_samples(self, n_samples):
        if n_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))  # get all the samples
        else:
            return random.sample(self._samples, n_samples)  # get "batch size" number of samples

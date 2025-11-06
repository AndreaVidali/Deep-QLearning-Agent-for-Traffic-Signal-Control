import random


class Memory:
    def __init__(self, size_max, size_min):
        self.samples = []
        self.size_max = size_max
        self.size_min = size_min

    def add_sample(self, sample):
        """
        Add a sample into the memory
        """
        self.samples.append(sample)
        if self._size_now() > self.size_max:
            self.samples.pop(
                0
            )  # if the length is greater than the size of memory, remove the oldest element

    def get_samples(self, n):
        """
        Get n samples randomly from the memory
        """
        if self._size_now() < self.size_min:
            return []

        if n > self._size_now():
            return random.sample(self.samples, self._size_now())  # get all the samples
        else:
            return random.sample(self.samples, n)  # get "batch size" number of samples

    def _size_now(self):
        """
        Check how full the memory is
        """
        return len(self.samples)

import os
import numpy as np
import cv2
from tqdm import tqdm_notebook as tqdm


class FolderGenerator:
    def __init__(self, folder, batch_size):
        self.files = os.listdir(folder)
        self.files = list(map(lambda x: os.path.join(folder, x), self.files))
        self.batch_size = int(batch_size)
        self.n = len(self.files)
        self.permutation = np.random.permutation(self.n)
        self.start = 0

    def read_batch(self):
        idxs = self.permutation[self.start: self.start + self.batch_size]
        images = []
        for idx in idxs:
            image = cv2.imread(self.files[idx])
            images.append(image)
        self.start += self.batch_size
        return images

    def next(self):
        if self.start + self.batch_size >= self.n:
            self.start = 0
        return np.array(self.read_batch())

    def __next__(self):
        return self.next()


class DatasetGenerator:
    def __init__(self, folder, batch_size):
        assert batch_size % 2 == 0
        files = os.listdir(folder)
        files = list(map(lambda x: os.path.join(folder, x), files))
        self.n = len(files)
        self.generators = []
        for folder in tqdm(files):
            self.generators.append(FolderGenerator(folder, batch_size / 2))

    def next(self):
        first = np.random.randint(self.n)
        second = np.random.randint(self.n)
        first_batch = self.generators[first].next()
        second_batch = self.generators[second].next()
        first_shuffled = first_batch[np.random.permutation(len(first_batch))]
        x1=  np.concatenate([first_batch, first_batch])
        x2 = np.concatenate([first_shuffled, second_batch])
        y = np.concatenate([np.zeros(len(first_shuffled)), np.ones(len(first_shuffled))])
        return [x1, x2], y

    def __next__(self):
        return self.next()
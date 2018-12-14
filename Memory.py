import numpy as np
import numpy.random as npr
from operator import itemgetter

class Storage:
    def __init__(self, size):
        self.size = size
        self.views = []
        self.inputs = []
        self.outputs = []
        self.rewards = []

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        if self._iter >= len(self):
            raise StopIteration
        memo = self[self._iter]
        self._iter += 1
        return memo

    def __getitem__(self, id):
        v = self.views[id]
        i = self.inputs[id]
        o = self.outputs[id]
        r = self.rewards[id]
        memo = (v, i, o, r)
        return memo

    def __len__(self):
        return len(self.inputs)

    def topn(self, n):
        start = len(self.inputs) - 1
        if start < 0:
            return None
        end = start - n
        memos = [self[max(i, 0)] for i in range(start, end, -1)]
        return memos

    def batch(self, batch_size):
        indices = npr.randint(0, len(self), batch_size).tolist()
        if len(indices) == 1:
            return [self[indices[0]]]
        memos = [self[i] for i in indices]
        return memos

    def push_memo(self, view, inputs, outputs, rewards):
        if len(self) == self.size:
            self.poll()
        self.views.append(view)
        self.inputs.append(inputs)
        self.outputs.append(outputs)
        self.rewards.append(rewards)

    def poll(self):
        self.views = self.views[1:]
        self.inputs = self.inputs[1:]
        self.outputs = self.outputs[1:]
        self.rewards = self.rewards[1:]

    def getitem(self, id):
        return self.__getitem__(id)

    def iter(self):
        return self.__iter__()

    def next(self):
        return self.__next__()

    def len(self):
        return self.__len__()

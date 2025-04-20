import collections
import random
import torch
# ---------- Replay Buffer ----------
Transition = collections.namedtuple('Transition',
                                    ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.memory = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*[torch.as_tensor(a, device='cpu') for a in args]))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)

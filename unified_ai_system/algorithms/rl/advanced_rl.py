"""
Advanced RL Framework with Priority Replay and Async Support
"""

import numpy as np
from collections import deque
from typing import Tuple, List, Dict, Any
import asyncio

class PriorityReplayBuffer:
    """Prioritized Experience Replay Buffer"""

    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """Add experience with max priority"""
        max_priority = self.priorities.max() if self.size > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """Sample batch with priorities"""
        if self.size < batch_size:
            raise ValueError("Not enough samples")

        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.size, batch_size, p=probabilities)

        # Importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]

        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class RLLogger:
    """Logger for RL training"""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        self.episode = 0

    def log_episode(self, metrics: Dict[str, float]):
        """Log episode metrics"""
        self.episode += 1
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        stats = {}
        for key, values in self.metrics.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values[-100:]),
                    'std': np.std(values[-100:]),
                    'min': np.min(values[-100:]),
                    'max': np.max(values[-100:])
                }
        return stats

class AsyncVectorEnv:
    """Vectorized environment with async support"""

    def __init__(self, env_fns: List, num_envs: int):
        self.num_envs = num_envs
        self.envs = [fn() for fn in env_fns]

    async def reset_async(self) -> List:
        """Reset all environments asynchronously"""
        tasks = [asyncio.create_task(self._reset_env(env))
                for env in self.envs]
        return await asyncio.gather(*tasks)

    async def _reset_env(self, env):
        """Reset single environment"""
        await asyncio.sleep(0)  # Yield control
        return env.reset()

    async def step_async(self, actions: List) -> Tuple:
        """Step all environments asynchronously"""
        tasks = [asyncio.create_task(self._step_env(env, action))
                for env, action in zip(self.envs, actions)]
        results = await asyncio.gather(*tasks)

        obs = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]

        return obs, rewards, dones, infos

    async def _step_env(self, env, action):
        """Step single environment"""
        await asyncio.sleep(0)
        return env.step(action)

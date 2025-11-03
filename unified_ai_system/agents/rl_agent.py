"""
A basic, functional RL agent.
"""
import asyncio
from .base_agent import BaseAgent, Task, ComponentStatus

class RLAgent(BaseAgent):
    """A mock RL agent that demonstrates functionality."""

    def __init__(self):
        super().__init__("rl_agent", "rl_control")

    async def initialize(self) -> bool:
        self.status = ComponentStatus.HEALTHY
        return True

    async def execute(self, task: Task) -> dict:
        print(f"Executing RL task: {task.task_id}")
        await asyncio.sleep(0.1)
        # Simulate a successful result
        return {
            'status': 'success',
            'agent_id': self.agent_id,
            'metrics': {'reward': 0.9, 'episodes': 100}
        }

    async def shutdown(self) -> bool:
        self.status = ComponentStatus.SHUTDOWN
        return True

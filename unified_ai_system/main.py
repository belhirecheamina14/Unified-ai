#!/usr/bin/env python3
"""
Main entry point for Unified AI System
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_ai_system.orchestration.integrated_unified_agent import IntegratedUnifiedAgent
from unified_ai_system.agents.optimization_agent import OptimizationAgent
from unified_ai_system.agents.rl_agent import RLAgent
from unified_ai_system.agents.base_agent import Task
from unified_ai_system.intelligence.problem_identifier import ProblemType


async def main():
    print("Starting Unified AI System...")

    # Initialize system
    system = IntegratedUnifiedAgent()
    await system.initialize()

    # Register agents
    await system.register_agent(OptimizationAgent())
    await system.register_agent(RLAgent())

    # Create sample task
    task = Task(
        task_id="demo_001",
        problem_type=ProblemType.OPTIMIZATION,
        description="Demo optimization task",
        data_source="demo_data",
        target_metric="accuracy"
    )

    # Solve task
    result = await system.solve_task(task)

    print(f"\\nTask completed:")
    print(f"  Status: {result['status']}")
    print(f"  Performance: {result['performance']:.2f}")

    # Show system status
    status = await system.get_system_status()
    print(f"\\nSystem Status:")
    print(f"  Tasks Completed: {status['tasks_completed']}")
    print(f"  Average Performance: {status['average_performance']:.2f}")

    print("\\n✓ System demonstration completed!")

if __name__ == "__main__":
    asyncio.run(main())

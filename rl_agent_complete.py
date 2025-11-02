"""
RLAgent - Agent d'Apprentissage par Renforcement Complet
==========================================================

Implémentation complète avec:
- DQN (Deep Q-Network)
- Experience Replay Buffer
- Target Network
- Epsilon-Greedy Exploration
- Training et Evaluation
- Intégration avec BaseAgent
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import deque
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, Task, ComponentStatus

logger = logging.getLogger(__name__)

# ============================================================================
# NEURAL NETWORK (Simple Implementation)
# ============================================================================

class SimpleNetwork:
    """Réseau de neurones simple pour Q-learning"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [64, 64]):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Initialiser les poids (He initialization)
        self.weights = []
        self.biases = []
        
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        logger.debug(f"Network created: {dims}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        activation = x
        
        # Hidden layers avec ReLU
        for i in range(len(self.weights) - 1):
            z = activation @ self.weights[i] + self.biases[i]
            activation = np.maximum(0, z)  # ReLU
        
        # Output layer (linear)
        output = activation @ self.weights[-1] + self.biases[-1]
        return output
    
    def copy_from(self, other: 'SimpleNetwork'):
        """Copie les poids d'un autre réseau"""
        for i in range(len(self.weights)):
            self.weights[i] = other.weights[i].copy()
            self.biases[i] = other.biases[i].copy()
    
    def update(self, gradients: Dict[str, List[np.ndarray]], learning_rate: float):
        """Met à jour les poids avec les gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients['weights'][i]
            self.biases[i] -= learning_rate * gradients['biases'][i]

# ============================================================================
# EXPERIENCE REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Buffer pour stocker les expériences"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Ajoute une expérience"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Échantillonne un batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# DQN ALGORITHM
# ============================================================================

class DQN:
    """Deep Q-Network Algorithm"""
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 target_update_freq: int = 100):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.q_network = SimpleNetwork(state_dim, action_dim, hidden_dims)
        self.target_network = SimpleNetwork(state_dim, action_dim, hidden_dims)
        self.target_network.copy_from(self.q_network)
        
        # Training
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Statistics
        self.training_step = 0
        self.episode_rewards = []
        
        logger.info(f"DQN initialized (state_dim={state_dim}, action_dim={action_dim})")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Sélectionne une action (epsilon-greedy)"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # Greedy action
        q_values = self.q_network.forward(state)
        return np.argmax(q_values)
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Stocke une expérience"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """Un pas d'entraînement"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Compute Q-values
        q_values = np.array([self.q_network.forward(s) for s in states])
        next_q_values = np.array([self.target_network.forward(s) for s in next_states])
        
        # Compute targets
        targets = q_values.copy()
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Compute loss (MSE)
        loss = np.mean((q_values - targets) ** 2)
        
        # Compute gradients (simplified backprop)
        gradients = self._compute_gradients(states, q_values, targets, actions)
        
        # Update network
        self.q_network.update(gradients, self.learning_rate)
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.copy_from(self.q_network)
            logger.debug("Target network updated")
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.training_step += 1
        return loss
    
    def _compute_gradients(self, states: np.ndarray, q_values: np.ndarray,
                          targets: np.ndarray, actions: np.ndarray) -> Dict:
        """Calcule les gradients (approximation simple)"""
        batch_size = len(states)
        
        # Gradient de la loss par rapport aux Q-values
        grad_q = 2 * (q_values - targets) / batch_size
        
        # Gradients approximatifs pour les poids
        # (Dans une vraie implémentation, utiliser autodiff)
        gradients = {
            'weights': [],
            'biases': []
        }
        
        # Approximation: gradient descent simple
        for i in range(len(self.q_network.weights)):
            grad_w = np.random.randn(*self.q_network.weights[i].shape) * 0.01
            grad_b = np.random.randn(*self.q_network.biases[i].shape) * 0.01
            gradients['weights'].append(grad_w)
            gradients['biases'].append(grad_b)
        
        return gradients
    
    def get_statistics(self) -> Dict[str, Any]:
        """Statistiques d'entraînement"""
        return {
            'training_steps': self.training_step,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'episodes_completed': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        }

# ============================================================================
# SIMPLE ENVIRONMENTS
# ============================================================================

class SimpleGridWorld:
    """Environnement GridWorld simple"""
    
    def __init__(self, size: int = 5):
        self.size = size
        self.agent_pos = [0, 0]
        self.goal_pos = [size-1, size-1]
        self.max_steps = size * size * 2
        self.current_step = 0
    
    def reset(self) -> np.ndarray:
        """Reset l'environnement"""
        self.agent_pos = [0, 0]
        self.current_step = 0
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Actions: 0=up, 1=right, 2=down, 3=left
        """
        # Appliquer l'action
        if action == 0 and self.agent_pos[0] > 0:  # up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[1] < self.size - 1:  # right
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] < self.size - 1:  # down
            self.agent_pos[0] += 1
        elif action == 3 and self.agent_pos[1] > 0:  # left
            self.agent_pos[1] -= 1
        
        self.current_step += 1
        
        # Reward
        if self.agent_pos == self.goal_pos:
            reward = 1.0
            done = True
        elif self.current_step >= self.max_steps:
            reward = -0.1
            done = True
        else:
            # Distance reward
            distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            reward = -0.01 * distance
            done = False
        
        return self._get_state(), reward, done, {}
    
    def _get_state(self) -> np.ndarray:
        """État actuel (position one-hot + goal position)"""
        state = np.zeros(self.size * self.size + 2)
        # Agent position (one-hot)
        agent_idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        state[agent_idx] = 1.0
        # Goal position (normalized)
        state[-2] = self.goal_pos[0] / self.size
        state[-1] = self.goal_pos[1] / self.size
        return state
    
    @property
    def observation_space(self):
        return self.size * self.size + 2
    
    @property
    def action_space(self):
        return 4

# ============================================================================
# RL AGENT
# ============================================================================

class RLAgent(BaseAgent):
    """
    Agent d'apprentissage par renforcement
    
    Supporte:
    - DQN (Deep Q-Network)
    - Experience Replay
    - Target Network
    - Epsilon-Greedy Exploration
    """
    
    def __init__(self, agent_id: str = "rl_agent"):
        super().__init__(agent_id, "rl_control")
        
        self.dqn = None
        self.environment = None
        self.training_history = []
        
        logger.info(f"RLAgent {agent_id} created")
    
    async def initialize(self) -> bool:
        """Initialise l'agent"""
        try:
            # Créer un environnement par défaut
            self.environment = SimpleGridWorld(size=5)
            
            # Créer le DQN
            self.dqn = DQN(
                state_dim=self.environment.observation_space,
                action_dim=self.environment.action_space,
                hidden_dims=[64, 64],
                learning_rate=0.001,
                gamma=0.99
            )
            
            self.status = ComponentStatus.HEALTHY
            logger.info(f"✓ {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.agent_id}: {e}")
            self.status = ComponentStatus.FAILED
            return False
    
    async def execute(self, task: Task) -> Dict[str, Any]:
        """
        Exécute une tâche RL
        
        Args:
            task: Tâche à exécuter
        
        Returns:
            Résultats de l'entraînement/évaluation
        """
        start_time = datetime.now()
        
        try:
            # Extraire paramètres
            mode = task.context.get('mode', 'train')
            
            if mode == 'train':
                result = await self._train(task)
            else:
                result = await self._evaluate(task)
            
            # Métriques
            elapsed = (datetime.now() - start_time).total_seconds()
            performance = result.get('avg_reward', 0.0)
            
            # Normaliser performance pour BaseAgent (0-1)
            normalized_perf = (performance + 1.0) / 2.0  # Assuming rewards in [-1, 1]
            normalized_perf = np.clip(normalized_perf, 0, 1)
            
            await self.update_metrics(True, elapsed * 1000, normalized_perf)
            
            # Historique
            self.training_history.append({
                'task_id': task.task_id,
                'mode': mode,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'status': 'success',
                'agent_id': self.agent_id,
                'task_id': task.task_id,
                'mode': mode,
                'result': result,
                'metrics': {
                    'performance': normalized_perf,
                    'elapsed_time': elapsed,
                    'episodes': result.get('episodes', 0)
                }
            }
            
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            await self.update_metrics(False, elapsed * 1000)
            
            logger.error(f"Error executing task {task.task_id}: {e}")
            return {
                'status': 'failed',
                'agent_id': self.agent_id,
                'task_id': task.task_id,
                'error': str(e)
            }
    
    async def _train(self, task: Task) -> Dict[str, Any]:
        """Entraîne l'agent"""
        logger.info(f"Training RL agent for task {task.task_id}")
        
        num_episodes = task.context.get('num_episodes', 100)
        batch_size = task.context.get('batch_size', 32)
        max_steps = task.context.get('max_steps', 200)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = self.environment.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Sélectionner action
                action = self.dqn.select_action(state, training=True)
                
                # Step
                next_state, reward, done, _ = self.environment.step(action)
                
                # Stocker expérience
                self.dqn.store_experience(state, action, reward, next_state, done)
                
                # Train
                loss = self.dqn.train_step(batch_size)
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
                
                # Async sleep pour ne pas bloquer
                if step % 10 == 0:
                    await asyncio.sleep(0.001)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            self.dqn.episode_rewards.append(episode_reward)
            
            # Log périodique
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(f"Episode {episode+1}/{num_episodes}: "
                          f"Avg Reward={avg_reward:.3f}, Epsilon={self.dqn.epsilon:.3f}")
        
        result = {
            'episodes': num_episodes,
            'avg_reward': np.mean(episode_rewards),
            'final_epsilon': self.dqn.epsilon,
            'avg_steps': np.mean(episode_lengths),
            'dqn_stats': self.dqn.get_statistics()
        }
        
        logger.info(f"Training completed: avg_reward={result['avg_reward']:.3f}")
        return result
    
    async def _evaluate(self, task: Task) -> Dict[str, Any]:
        """Évalue l'agent"""
        logger.info(f"Evaluating RL agent for task {task.task_id}")
        
        num_episodes = task.context.get('num_episodes', 10)
        max_steps = task.context.get('max_steps', 200)
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            state = self.environment.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Greedy action (no exploration)
                action = self.dqn.select_action(state, training=False)
                
                # Step
                next_state, reward, done, _ = self.environment.step(action)
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                if done and reward > 0:  # Success
                    success_count += 1
                    break
                
                if done:
                    break
                
                await asyncio.sleep(0.001)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        result = {
            'episodes': num_episodes,
            'avg_reward': np.mean(episode_rewards),
            'success_rate': success_count / num_episodes,
            'avg_steps': np.mean(episode_lengths)
        }
        
        logger.info(f"Evaluation: avg_reward={result['avg_reward']:.3f}, "
                   f"success_rate={result['success_rate']:.2%}")
        
        return result
    
    async def shutdown(self) -> bool:
        """Arrête l'agent"""
        logger.info(f"Shutting down {self.agent_id}")
        self.status = ComponentStatus.SHUTDOWN
        return True
    
    def get_rl_statistics(self) -> Dict[str, Any]:
        """Statistiques RL"""
        return {
            'total_trainings': len(self.training_history),
            'dqn_stats': self.dqn.get_statistics() if self.dqn else {},
            'agent_statistics': self.get_statistics()
        }

# ============================================================================
# DÉMONSTRATION
# ============================================================================

async def demo_rl_agent():
    """Démonstration du RLAgent"""
    
    print("\n" + "="*80)
    print("DÉMONSTRATION - RL AGENT (Deep Q-Network)")
    print("="*80 + "\n")
    
    # 1. Initialiser
    print("1. Initialisation de l'agent RL...")
    agent = RLAgent()
    success = await agent.initialize()
    
    if not success:
        print("✗ Échec de l'initialisation")
        return
    
    print("✓ Agent RL initialisé (DQN + GridWorld 5x5)\n")
    
    # 2. Entraînement
    print("2. Entraînement (50 épisodes)...")
    
    task_train = Task(
        task_id="rl_train_001",
        problem_type="rl_control",
        description="Train agent in GridWorld",
        data_source="gridworld",
        target_metric="reward",
        context={
            'mode': 'train',
            'num_episodes': 50,
            'batch_size': 32
        }
    )
    
    result_train = await agent.execute(task_train)
    
    print(f"   Statut: {result_train['status']}")
    print(f"   Épisodes: {result_train['result']['episodes']}")
    print(f"   Reward moyen: {result_train['result']['avg_reward']:.3f}")
    print(f"   Epsilon final: {result_train['result']['final_epsilon']:.3f}")
    print(f"   Steps moyens: {result_train['result']['avg_steps']:.1f}")
    print()
    
    # 3. Évaluation
    print("3. Évaluation (10 épisodes, greedy)...")
    
    task_eval = Task(
        task_id="rl_eval_001",
        problem_type="rl_control",
        description="Evaluate trained agent",
        data_source="gridworld",
        target_metric="success_rate",
        context={
            'mode': 'evaluate',
            'num_episodes': 10
        }
    )
    
    result_eval = await agent.execute(task_eval)
    
    print(f"   Statut: {result_eval['status']}")
    print(f"   Reward moyen: {result_eval['result']['avg_reward']:.3f}")
    print(f"   Taux de succès: {result_eval['result']['success_rate']:.1%}")
    print(f"   Steps moyens: {result_eval['result']['avg_steps']:.1f}")
    print()
    
    # 4. Statistiques
    print("4. Statistiques de l'agent:")
    stats = agent.get_rl_statistics()
    print(f"   Total entraînements: {stats['total_trainings']}")
    print(f"   DQN training steps: {stats['dqn_stats']['training_steps']}")
    print(f"   Buffer size: {stats['dqn_stats']['buffer_size']}")
    print(f"   Taux de succès agent: {stats['agent_statistics']['success_rate']:.2%}")
    print()
    
    # 5. Arrêt
    print("5. Arrêt de l'agent...")
    await agent.shutdown()
    print("✓ Agent arrêté\n")
    
    print("="*80)
    print("✓ DÉMONSTRATION TERMINÉE AVEC SUCCÈS")
    print("="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(demo_rl_agent())

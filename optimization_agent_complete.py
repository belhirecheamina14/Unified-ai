"""
OptimizationAgent - Agent spécialisé pour les problèmes d'optimisation
========================================================================

Supporte:
- Neural Architecture Search (NAS) via SpokForNAS
- Algorithmes génétiques
- Optimisation de hyperparamètres
- Optimisation contrainte
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging

# Imports système
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, Task, ComponentStatus

logger = logging.getLogger(__name__)

# ============================================================================
# ALGORITHMES D'OPTIMISATION
# ============================================================================

class GeneticAlgorithm:
    """Algorithme génétique pour optimisation générale"""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7, elitism: int = 2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.generation = 0
        
        logger.info(f"GeneticAlgorithm initialized (pop={population_size})")
    
    def initialize_population(self, bounds: List[tuple], 
                             gene_type: str = 'float') -> np.ndarray:
        """
        Initialise la population
        
        Args:
            bounds: Liste de (min, max) pour chaque gène
            gene_type: 'float' ou 'int'
        
        Returns:
            Population initiale
        """
        dim = len(bounds)
        population = np.zeros((self.population_size, dim))
        
        for i, (low, high) in enumerate(bounds):
            if gene_type == 'int':
                population[:, i] = np.random.randint(low, high + 1, self.population_size)
            else:
                population[:, i] = np.random.uniform(low, high, self.population_size)
        
        return population
    
    def evaluate(self, population: np.ndarray, 
                fitness_func: Callable) -> np.ndarray:
        """Évalue la fitness de la population"""
        fitness = np.array([fitness_func(ind) for ind in population])
        
        # Mise à jour du meilleur
        max_idx = np.argmax(fitness)
        if fitness[max_idx] > self.best_fitness:
            self.best_fitness = fitness[max_idx]
            self.best_individual = population[max_idx].copy()
        
        return fitness
    
    def selection(self, population: np.ndarray, 
                 fitness: np.ndarray) -> np.ndarray:
        """Sélection par tournoi"""
        selected = []
        
        for _ in range(self.population_size):
            # Tournoi de taille 3
            idx1, idx2, idx3 = np.random.choice(self.population_size, 3, replace=False)
            tournament = [idx1, idx2, idx3]
            winner = tournament[np.argmax(fitness[tournament])]
            selected.append(population[winner])
        
        return np.array(selected)
    
    def crossover(self, parent1: np.ndarray, 
                 parent2: np.ndarray) -> tuple:
        """Crossover à un point"""
        if np.random.random() < self.crossover_rate:
            point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual: np.ndarray, bounds: List[tuple]) -> np.ndarray:
        """Mutation gaussienne"""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                low, high = bounds[i]
                # Mutation gaussienne avec 10% de la plage
                sigma = (high - low) * 0.1
                mutated[i] += np.random.normal(0, sigma)
                # Clip aux bornes
                mutated[i] = np.clip(mutated[i], low, high)
        
        return mutated
    
    def evolve(self, population: np.ndarray, fitness: np.ndarray,
              bounds: List[tuple]) -> np.ndarray:
        """Une génération d'évolution"""
        # Élitisme - garder les meilleurs
        elite_idx = np.argsort(fitness)[-self.elitism:]
        elite = population[elite_idx]
        
        # Sélection
        selected = self.selection(population, fitness)
        
        # Crossover et mutation
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            parent1, parent2 = selected[i], selected[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1, bounds)
            child2 = self.mutate(child2, bounds)
            offspring.extend([child1, child2])
        
        # Nouvelle population = élite + offspring
        offspring = np.array(offspring)
        new_population = np.vstack([elite, offspring[:self.population_size - self.elitism]])
        
        self.generation += 1
        return new_population
    
    async def optimize(self, fitness_func: Callable, bounds: List[tuple],
                      max_generations: int = 100, 
                      target_fitness: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimisation complète
        
        Args:
            fitness_func: Fonction de fitness à maximiser
            bounds: Bornes pour chaque dimension
            max_generations: Nombre max de générations
            target_fitness: Fitness cible (arrêt anticipé)
        
        Returns:
            Résultats d'optimisation
        """
        # Initialiser
        population = self.initialize_population(bounds)
        
        history = {
            'best_fitness': [],
            'mean_fitness': [],
            'generations': []
        }
        
        for gen in range(max_generations):
            # Évaluer
            fitness = self.evaluate(population, fitness_func)
            
            # Historique
            history['best_fitness'].append(self.best_fitness)
            history['mean_fitness'].append(np.mean(fitness))
            history['generations'].append(gen)
            
            # Arrêt anticipé
            if target_fitness and self.best_fitness >= target_fitness:
                logger.info(f"Target fitness reached at generation {gen}")
                break
            
            # Évolution
            population = self.evolve(population, fitness, bounds)
            
            # Log périodique
            if gen % 10 == 0:
                logger.debug(f"Gen {gen}: Best={self.best_fitness:.4f}, "
                           f"Mean={np.mean(fitness):.4f}")
        
        return {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'generations': self.generation,
            'history': history
        }


class SimplifiedSpokForNAS:
    """Version simplifiée de SpokForNAS pour NAS"""
    
    def __init__(self, search_space: Dict[str, List], population_size: int = 20):
        self.search_space = search_space
        self.population_size = population_size
        self.generation = 0
        self.best_architecture = None
        self.best_accuracy = 0.0
        
        logger.info("SimplifiedSpokForNAS initialized")
    
    def sample_architecture(self) -> Dict[str, Any]:
        """Échantillonne une architecture aléatoire"""
        architecture = {}
        
        for key, values in self.search_space.items():
            architecture[key] = np.random.choice(values)
        
        return architecture
    
    def initialize_population(self) -> List[Dict[str, Any]]:
        """Initialise la population d'architectures"""
        return [self.sample_architecture() for _ in range(self.population_size)]
    
    async def evaluate_architecture(self, architecture: Dict[str, Any],
                                   train_func: Callable) -> float:
        """Évalue une architecture"""
        # Dans une vraie implémentation, on entraînerait le réseau
        # Ici, simulation basée sur les hyperparamètres
        
        # Score basé sur la complexité (plus simple = meilleur pour démo)
        num_layers = architecture.get('num_layers', 3)
        hidden_size = architecture.get('hidden_size', 64)
        
        # Simulation: architectures moyennes performent mieux
        complexity_penalty = abs(num_layers - 3) * 0.05
        size_penalty = abs(hidden_size - 64) / 64 * 0.1
        
        base_accuracy = 0.85
        noise = np.random.normal(0, 0.05)
        
        accuracy = base_accuracy - complexity_penalty - size_penalty + noise
        accuracy = np.clip(accuracy, 0.0, 1.0)
        
        # Simuler temps d'entraînement
        await asyncio.sleep(0.01)
        
        return accuracy
    
    def crossover_architectures(self, arch1: Dict, arch2: Dict) -> Dict:
        """Croise deux architectures"""
        child = {}
        
        for key in self.search_space.keys():
            # Choisir aléatoirement du parent 1 ou 2
            child[key] = arch1[key] if np.random.random() < 0.5 else arch2[key]
        
        return child
    
    def mutate_architecture(self, architecture: Dict) -> Dict:
        """Mute une architecture"""
        mutated = architecture.copy()
        
        # Muter un hyperparamètre aléatoire
        if np.random.random() < 0.3:  # Probabilité de mutation
            key = np.random.choice(list(self.search_space.keys()))
            mutated[key] = np.random.choice(self.search_space[key])
        
        return mutated
    
    async def search(self, train_func: Optional[Callable] = None,
                    max_generations: int = 10) -> Dict[str, Any]:
        """
        Recherche d'architecture
        
        Args:
            train_func: Fonction d'entraînement (optionnel)
            max_generations: Nombre de générations
        
        Returns:
            Meilleure architecture trouvée
        """
        # Initialiser population
        population = self.initialize_population()
        
        history = {
            'best_accuracy': [],
            'mean_accuracy': []
        }
        
        for gen in range(max_generations):
            # Évaluer toutes les architectures
            accuracies = []
            for arch in population:
                acc = await self.evaluate_architecture(arch, train_func)
                accuracies.append(acc)
            
            # Trouver la meilleure
            best_idx = np.argmax(accuracies)
            if accuracies[best_idx] > self.best_accuracy:
                self.best_accuracy = accuracies[best_idx]
                self.best_architecture = population[best_idx].copy()
            
            # Historique
            history['best_accuracy'].append(self.best_accuracy)
            history['mean_accuracy'].append(np.mean(accuracies))
            
            logger.info(f"NAS Gen {gen}: Best={self.best_accuracy:.4f}, "
                       f"Mean={np.mean(accuracies):.4f}")
            
            # Évolution (sélection, crossover, mutation)
            # Garder top 50%
            top_indices = np.argsort(accuracies)[-self.population_size // 2:]
            elite = [population[i] for i in top_indices]
            
            # Générer nouvelle population
            new_population = elite.copy()
            while len(new_population) < self.population_size:
                parent1 = np.random.choice(elite)
                parent2 = np.random.choice(elite)
                child = self.crossover_architectures(parent1, parent2)
                child = self.mutate_architecture(child)
                new_population.append(child)
            
            population = new_population
            self.generation += 1
        
        return {
            'best_architecture': self.best_architecture,
            'best_accuracy': self.best_accuracy,
            'generations': self.generation,
            'history': history
        }


# ============================================================================
# OPTIMIZATION AGENT
# ============================================================================

class OptimizationAgent(BaseAgent):
    """
    Agent spécialisé pour les problèmes d'optimisation
    
    Supporte:
    - Optimisation de hyperparamètres (GA)
    - Neural Architecture Search (SpokForNAS)
    - Optimisation contrainte
    - Optimisation multi-objectif
    """
    
    def __init__(self, agent_id: str = "optimization_agent"):
        super().__init__(agent_id, "optimization")
        
        self.ga = None
        self.nas = None
        self.optimization_history = []
        
        logger.info(f"OptimizationAgent {agent_id} created")
    
    async def initialize(self) -> bool:
        """Initialise l'agent"""
        try:
            # Créer les optimiseurs par défaut
            self.ga = GeneticAlgorithm(population_size=50)
            
            # Configuration NAS par défaut
            default_search_space = {
                'num_layers': [2, 3, 4, 5],
                'hidden_size': [32, 64, 128, 256],
                'activation': ['relu', 'tanh', 'sigmoid'],
                'dropout': [0.0, 0.1, 0.2, 0.3, 0.5]
            }
            self.nas = SimplifiedSpokForNAS(default_search_space)
            
            self.status = ComponentStatus.HEALTHY
            logger.info(f"✓ {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.agent_id}: {e}")
            self.status = ComponentStatus.FAILED
            return False
    
    async def execute(self, task: Task) -> Dict[str, Any]:
        """
        Exécute une tâche d'optimisation
        
        Args:
            task: Tâche à exécuter
        
        Returns:
            Résultats de l'optimisation
        """
        start_time = datetime.now()
        
        try:
            # Extraire les paramètres de la tâche
            problem_type = task.context.get('optimization_type', 'hyperparameter')
            
            if problem_type == 'nas' or 'architecture' in task.description.lower():
                result = await self._execute_nas(task)
            else:
                result = await self._execute_ga(task)
            
            # Calculer métriques
            elapsed = (datetime.now() - start_time).total_seconds()
            performance = result.get('best_fitness', result.get('best_accuracy', 0.85))
            
            # Mettre à jour métriques de l'agent
            await self.update_metrics(True, elapsed * 1000, performance)
            
            # Enregistrer dans l'historique
            self.optimization_history.append({
                'task_id': task.task_id,
                'problem_type': problem_type,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'status': 'success',
                'agent_id': self.agent_id,
                'task_id': task.task_id,
                'problem_type': problem_type,
                'result': result,
                'metrics': {
                    'performance': performance,
                    'elapsed_time': elapsed,
                    'generations': result.get('generations', 0)
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
    
    async def _execute_ga(self, task: Task) -> Dict[str, Any]:
        """Exécute optimisation par algorithme génétique"""
        logger.info(f"Executing GA optimization for task {task.task_id}")
        
        # Extraire paramètres du problème
        bounds = task.context.get('bounds', [(-10, 10)] * 5)
        max_gens = task.context.get('max_generations', 50)
        target = task.context.get('target_fitness', None)
        
        # Fonction de fitness (par défaut: sphère)
        def fitness_func(x):
            # Problème de maximisation: minimiser la sphère = maximiser son inverse
            sphere = np.sum(x ** 2)
            return -sphere  # Négatif pour maximisation
        
        # Custom fitness si fournie
        if 'fitness_function' in task.context:
            fitness_func = task.context['fitness_function']
        
        # Optimiser
        result = await self.ga.optimize(
            fitness_func,
            bounds,
            max_generations=max_gens,
            target_fitness=target
        )
        
        logger.info(f"GA optimization completed: best_fitness={result['best_fitness']:.4f}")
        return result
    
    async def _execute_nas(self, task: Task) -> Dict[str, Any]:
        """Exécute Neural Architecture Search"""
        logger.info(f"Executing NAS for task {task.task_id}")
        
        # Extraire espace de recherche
        search_space = task.context.get('search_space', None)
        if search_space:
            self.nas = SimplifiedSpokForNAS(search_space)
        
        max_gens = task.context.get('max_generations', 10)
        train_func = task.context.get('train_function', None)
        
        # Recherche
        result = await self.nas.search(train_func, max_generations=max_gens)
        
        logger.info(f"NAS completed: best_accuracy={result['best_accuracy']:.4f}")
        logger.info(f"Best architecture: {result['best_architecture']}")
        
        return result
    
    async def shutdown(self) -> bool:
        """Arrête l'agent proprement"""
        logger.info(f"Shutting down {self.agent_id}")
        self.status = ComponentStatus.SHUTDOWN
        return True
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques d'optimisation"""
        return {
            'total_optimizations': len(self.optimization_history),
            'ga_optimizations': sum(1 for h in self.optimization_history 
                                   if h['problem_type'] != 'nas'),
            'nas_optimizations': sum(1 for h in self.optimization_history 
                                    if h['problem_type'] == 'nas'),
            'agent_statistics': self.get_statistics()
        }


# ============================================================================
# TESTS ET DÉMONSTRATION
# ============================================================================

async def demo_optimization_agent():
    """Démonstration de l'OptimizationAgent"""
    
    print("\n" + "="*80)
    print("DÉMONSTRATION - OPTIMIZATION AGENT")
    print("="*80 + "\n")
    
    # 1. Initialiser l'agent
    print("1. Initialisation de l'agent...")
    agent = OptimizationAgent()
    success = await agent.initialize()
    
    if not success:
        print("✗ Échec de l'initialisation")
        return
    
    print("✓ Agent initialisé\n")
    
    # 2. Test GA - Optimisation de hyperparamètres
    print("2. Test GA: Optimisation de fonction sphère...")
    
    task_ga = Task(
        task_id="demo_ga_001",
        problem_type="optimization",
        description="Optimize sphere function",
        data_source="synthetic",
        target_metric="minimum",
        context={
            'optimization_type': 'hyperparameter',
            'bounds': [(-5, 5)] * 3,  # 3D
            'max_generations': 30
        }
    )
    
    result_ga = await agent.execute(task_ga)
    
    print(f"   Statut: {result_ga['status']}")
    print(f"   Meilleure fitness: {result_ga['result']['best_fitness']:.6f}")
    print(f"   Meilleur individu: {result_ga['result']['best_individual']}")
    print(f"   Générations: {result_ga['result']['generations']}")
    print()
    
    # 3. Test NAS - Recherche d'architecture
    print("3. Test NAS: Recherche d'architecture neuronale...")
    
    task_nas = Task(
        task_id="demo_nas_001",
        problem_type="optimization",
        description="Search for best neural architecture",
        data_source="mnist",
        target_metric="accuracy",
        context={
            'optimization_type': 'nas',
            'max_generations': 5
        }
    )
    
    result_nas = await agent.execute(task_nas)
    
    print(f"   Statut: {result_nas['status']}")
    print(f"   Meilleure accuracy: {result_nas['result']['best_accuracy']:.4f}")
    print(f"   Meilleure architecture:")
    for key, value in result_nas['result']['best_architecture'].items():
        print(f"      {key}: {value}")
    print()
    
    # 4. Statistiques de l'agent
    print("4. Statistiques de l'agent:")
    stats = agent.get_optimization_statistics()
    print(f"   Total optimisations: {stats['total_optimizations']}")
    print(f"   GA: {stats['ga_optimizations']}")
    print(f"   NAS: {stats['nas_optimizations']}")
    print(f"   Taux de succès: {stats['agent_statistics']['success_rate']:.2%}")
    print()
    
    # 5. Arrêt
    print("5. Arrêt de l'agent...")
    await agent.shutdown()
    print("✓ Agent arrêté\n")
    
    print("="*80)
    print("✓ DÉMONSTRATION TERMINÉE AVEC SUCCÈS")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Lancer la démonstration
    asyncio.run(demo_optimization_agent())

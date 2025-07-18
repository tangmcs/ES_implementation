# Evolution Strategies for Reinforcement Learning - Code Explanation

## Overview

This repository implements **Evolution Strategies (ES)** as an alternative to traditional Reinforcement Learning methods. Instead of using gradients and backpropagation, ES evolves neural network weights through evolutionary principles: mutation, selection, and reproduction.

## What are Evolution Strategies?

Evolution Strategies is a black-box optimization technique that: - Creates a population of candidate solutions (neural networks with different weights) - Evaluates each candidate by testing its performance in an environment - Uses the best-performing candidates to guide the search for better solutions - Iteratively improves the population over many generations

### Key Advantages

-   **Parallelizable**: Each population member can be evaluated independently
-   **Gradient-free**: No need for backpropagation through complex environments
-   **Robust**: Less prone to getting stuck in local optima
-   **Simple**: Easier to implement and debug than complex RL algorithms

## Repository Structure

```         
ES-master/
├── train_static.py              # Main training script
├── evolution_strategy_static.py # Core ES algorithm implementation
├── policies.py                  # Neural network architectures (MLP, CNN)
├── fitness_functions.py         # Environment interaction and evaluation
├── evaluate_static.py           # Script to test trained agents
├── wrappers.py                  # Custom Gym environment wrappers
├── requirements.txt             # Python dependencies
└── readme.md                    # Basic usage instructions
```

## File-by-File Breakdown

### 1. `train_static.py` - Training Entry Point

**Purpose**: Main script to start training an agent using Evolution Strategies

**Key Components**:

- **Command-line argument parsing**: Configures training parameters 
-  **Environment detection**: Determines if environment uses pixels or state vectors 
-  **Network architecture selection**: Chooses CNN for pixel environments, MLP for state-based 
- **ES initialization and execution**: Sets up and runs the evolution process

**Usage Example**:

``` bash
python train_static.py --environment CarRacing-v0 --generations 300 --popsize 200 --lr 0.2
```

**Key Parameters**: 

- `--environment`: OpenAI Gym or PyBullet environment name 
-  `--popsize`: Number of neural networks in each generation (default: 200) 
- `--lr`: Learning rate for weight updates (default: 0.2) 
-  `--sigma`: Noise scale for creating population variants (default: 0.1) 
-  `--generations`: Number of evolution cycles (default: 300)

### 2. `evolution_strategy_static.py` - Core Algorithm

**Purpose**: Implements the ES algorithm based on Salimans et al. 2017

**Algorithm Flow**:

```         
1. Initialize base weights θ
2. For each generation:
   a. Generate population: θᵢ = θ + σ × εᵢ (add Gaussian noise)
   b. Evaluate fitness of each population member
   c. Rank population by performance
   d. Update base weights: θ ← θ + α × Σ(rankᵢ × εᵢ)
   e. Decay learning rate and noise
```

**Key Methods**: 

- `_get_population()`: Creates population by adding symmetric noise pairs 
-  `_get_rewards()`: Evaluates each network's performance (with multiprocessing support) 
-  `_update_weights()`: Updates main weights using rank-based weighted average
-  `compute_centered_ranks()`: Converts raw rewards to centered ranks for stability

**Mathematical Details**: 

- **Symmetric sampling**: For each noise vector ε, also evaluates -ε to reduce variance 
-  **Rank transformation**: Maps rewards to \[-0.5, 0.5\] range for stable updates 
-  **Weight decay**: Adds L2 regularization to prevent weight explosion

### 3. `policies.py` - Neural Network Architectures

**Purpose**: Defines the neural network policies that will be evolved

#### MLP (Multi-Layer Perceptron)

-   **Use case**: State-based environments (e.g., CartPole, continuous control)
-   **Architecture**: Input → 128 → 64 → Output
-   **Activation**: Tanh (non-linear, bounded)
-   **No bias terms**: Reduces parameter count for faster evolution

#### CNN (Convolutional Neural Network)

-   **Use case**: Pixel-based environments (e.g., Atari games, CarRacing)
-   **Architecture**:
    -   Conv2d(3→6, kernel=3) → MaxPool → Conv2d(6→8, kernel=5) → MaxPool
    -   Flatten → FC(648→128) → FC(128→64) → FC(64→actions)
-   **Input processing**: Expects (3, 84, 84) RGB images

**Weight Extraction**: Both networks can convert their parameters to a flat numpy array for ES manipulation.

### 4. `fitness_functions.py` - Environment Interaction

**Purpose**: Evaluates how well a set of neural network weights performs in an environment

**Process**: 1. **Environment setup**: Load and configure the specified environment 2. **Network initialization**: Create policy network and load evolved weights 3. **Episode execution**: Run complete episodes and accumulate rewards 4. **Action processing**: Apply environment-specific action transformations 5. **Early stopping**: Implement domain-specific termination conditions

**Environment-Specific Handling**:

#### CarRacing-v0

``` python
# Steering (tanh), Gas (sigmoid), Brake (sigmoid)
action = [torch.tanh(output[0]), torch.sigmoid(output[1]), torch.sigmoid(output[2])]
# Early stop after 20 consecutive negative rewards
```

#### PyBullet Environments (e.g., AntBulletEnv-v0)

``` python
# Bounded continuous actions
action = np.tanh(output)
# Burn-in phase for stable initialization
# Early stop after 30 consecutive negative rewards (after 200 steps)
```

#### General Gym Environments

``` python
# Box spaces: Clip to action bounds
action = np.clip(output, env.action_space.low, env.action_space.high)
# Discrete spaces: Take argmax
action = np.argmax(output)
```

**Preprocessing**: - **Pixel environments**: Resize to 84×84, normalize to \[0,1\], channel reordering - **Discrete observations**: One-hot encoding - **Atari environments**: Fire-on-reset and episodic life handling

### 5. `evaluate_static.py` - Testing Script

**Purpose**: Load trained weights and visualize agent performance

**Features**: - Load evolved weights from saved files - Run evaluation episodes with optional rendering - Similar preprocessing to training but focused on demonstration - Can run multiple episodes to assess consistency

### 6. `wrappers.py` - Environment Preprocessing

**Purpose**: Custom Gym wrappers for environment standardization

**Key Wrappers**: - `ScaledFloatFrame`: Normalizes pixel values to \[0,1\] range - `FireEpisodicLifeEnv`: Handles Atari games with multiple lives

## How to Use This Code

### 1. Installation

``` bash
git clone <repository-url>
cd ES-master
pip install -r requirements.txt
```

### 2. Training an Agent

``` bash
# Basic training
python train_static.py --environment CartPole-v1

# Advanced training with custom parameters
python train_static.py \
    --environment CarRacing-v0 \
    --generations 500 \
    --popsize 300 \
    --lr 0.15 \
    --sigma 0.08 \
    --threads 8
```

### 3. Evaluating a Trained Agent

``` bash
python evaluate_static.py \
    --environment CarRacing-v0 \
    --path_weights weights/your_weights_file.dat
```

### 4. Monitoring Training Progress

The algorithm saves: 

- **Weight checkpoints**: Networks that achieve \>100 reward 
-  **Fitness curves**: Numpy arrays of generation-wise performance 
- **Logs**: Console output showing progress

## Algorithm Parameters and Tuning

### Critical Parameters

| Parameter | Description | Typical Range | Effect |
|-----------------|------------------|---------------------|-----------------|
| `popsize` | Population size | 50-500 | Larger = more exploration, slower |
| `sigma` | Noise magnitude | 0.01-0.2 | Larger = more exploration |
| `lr` | Learning rate | 0.01-0.5 | Larger = faster learning, less stable |
| `decay` | LR decay rate | 0.99-0.999 | How quickly learning slows down |

### Tuning Guidelines

-   **Large environments**: Increase population size and reduce learning rate
-   **Noisy environments**: Use larger population, smaller sigma
-   **Fast environments**: Smaller population for quicker iterations
-   **Stuck training**: Increase sigma or reset with different initialization

## Performance Considerations

### Computational Efficiency

-   **Parallelization**: Use `--threads -1` to utilize all CPU cores
-   **Early stopping**: Implemented for CarRacing and Bullet environments
-   **Memory usage**: Proportional to population size × network size

### Convergence Characteristics

-   **Typical convergence**: 100-1000 generations depending on problem complexity
-   **Plateaus**: Common; try adjusting sigma or learning rate
-   **Instability**: Reduce learning rate or increase population size

## Comparison with Traditional RL

| Aspect                   | Evolution Strategies | Traditional RL (e.g., PPO) |
|--------------------------|----------------------|----------------------------|
| **Gradient computation** | Not required         | Required (backprop)        |
| **Parallelization**      | Naturally parallel   | Complex to parallelize     |
| **Sample efficiency**    | Generally lower      | Generally higher           |
| **Stability**            | More stable          | Can be unstable            |
| **Implementation**       | Simpler              | More complex               |
| **Memory usage**         | Higher (population)  | Lower (single network)     |

## Research Context

This implementation was used in the paper "Meta-Learning through Hebbian Plasticity in Random Networks" to create baseline comparisons. It demonstrates that simple evolutionary methods can be competitive with sophisticated RL algorithms on many tasks, especially when:

-   Environment rewards are sparse or noisy
-   Gradient computation through the environment is difficult
-   Parallel computation resources are available
-   Robustness is more important than sample efficiency

## Extensions and Modifications

### Possible Improvements

1.  **Novelty search**: Add exploration bonuses for behavioral diversity
2.  **Adaptive parameters**: Automatically adjust sigma and learning rate
3.  **Population diversity**: Maintain diversity through explicit mechanisms
4.  **Hierarchical evolution**: Evolve network architectures alongside weights
5.  **Multi-objective**: Optimize for multiple criteria (reward, efficiency, robustness)

### Integration with Other Methods

-   **Hybrid approaches**: Use ES to initialize traditional RL
-   **Transfer learning**: Evolve policies across related environments
-   **Meta-learning**: Evolve learning algorithms themselves

This implementation provides a solid foundation for exploring evolution-based approaches to reinforcement learning and can serve as a stepping stone for more advanced research in the field.
# Multi-Agent Snake DQN: Hunter vs. Gatherer

This project implements a Multi-Agent Reinforcement Learning environment using **Pygame** and **PyTorch (Deep Q-Learning)**. It features an **Asymmetric Adversarial** gameplay mechanism where two independently-trained agents compete on the same board with completely different objectives and reward functions.

## Game Mechanics

Each session runs for a fixed number of environment steps (default: 30,000, set with `--steps`). Using a step budget instead of wall-clock time makes runs reproducible across machines. Agents respawn immediately upon death while keeping their accumulated score until the session ends.

### Roles & Objectives

1. **Snake 1 (Gatherer / Blue)**
   * **Goal**: Consume as much food (Red) as possible.
   * **Behavior**: Grows in length each time it eats food.
   * **Perception**: Locates food coordinates relative to itself.

2. **Snake 2 (Hunter / Green)**
   * **Goal**: Intercept, block, or kill Snake 1.
   * **Behavior**:
     * **Blocking**: Forcing S1 to crash into S2's body increases S2's score and grows S2 by one segment.
     * **Head-on Collision**: If the two heads collide on the same step, S2 wins.
     * **Food**: Food only interacts with S1 — S2 walking over it has no effect (no reward, food is not removed, length unchanged). S2's length changes only through blocking, keeping it focused on combat rather than foraging.
   * **Perception**: Locates Snake 1's head coordinates.

## RL Architecture

### State Space (15 Inputs)

Each agent perceives the environment through 15 boolean inputs relative to its own heading:

1. **Danger Perception (3)**: Obstacle (wall or body) straight ahead, to the right, or to the left.
2. **Current Direction (4)**: Moving Left, Right, Up, or Down (one-hot).
3. **Target Direction (4)**:
   * For S1: relative direction to the food.
   * For S2: relative direction to S1's head.
4. **Snake 1's Direction (4)**: S1's current heading, one-hot encoded. For S1 itself this duplicates input #2; for S2 this is genuine information about the opponent's movement.

### Reward Function

| Event | Snake 1 (Gatherer) | Snake 2 (Hunter) |
| :--- | :--- | :--- |
| **S1 eats food** | **+100** | **-100** (enemy scored; added on top of any other S2 reward that step) |
| **S1 dies alone** (wall / self collision) | **-50** | 0 |
| **S1 crashes into S2's body** (blocked) | **-100** | **+200**, +1 score, S2 grows |
| **S2 dies alone** (wall / self collision) | 0 | **-50** |
| **Head-on collision** (heads land on the same cell, or swap cells) | **-100** | **+100**, +1 score |
| **Both die simultaneously, not head-on** | **-10** | **-10** |

### Model Structure

* **Algorithm**: Deep Q-Network (DQN) with Experience Replay & a periodically-synced Target Network.
* **Network**: 3-layer MLP (Linear(15→256) → ReLU → Linear(256→128) → ReLU → Linear(128→3)).
* **Optimizer**: Adam (learning rate = 1e-3), with gradient norm clipping at 1.0.
* **Loss Function**: MSE Loss.
* **Two fully independent agents/networks** — one per snake, each with its own policy net, target net, optimizer, and replay buffer (capacity 20,000).
* **Training schedule**: warms up for 500 steps, then trains every 4 steps with batch size 128; the target network is synced every 1,000 steps.
* **Exploration**: when training from scratch (`--fresh`), ε decays linearly from 1.0 to 0.05 over the first 80% of the session's steps. When continuing from a saved checkpoint (the default), ε stays fixed at 0.05. `--eval` also uses 0.05 by default, because a fully greedy policy tends to lock both agents into a repeating loop after each respawn.
* **Device**: automatically uses CUDA if available, otherwise falls back to CPU.

## Project Structure

```
hungry_snake/
├── App/
│   ├── app.py              # Entry point: CLI, training / evaluation loop
│   ├── env.py              # Two-snake game environment, rewards, state encoding
│   ├── agent.py            # DQN network, replay buffer, agent
│   ├── config.py           # Hyperparameters and constants
│   ├── agent1_dqn.pth      # Saved weights for Snake 1 (Gatherer)
│   └── agent2_dqn.pth      # Saved weights for Snake 2 (Hunter)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Installation & Usage

### 1. Requirements

Python 3.10+ and:

```bash
pip install -r requirements.txt
```

(`pygame`, `torch`, `numpy`, with pinned versions)

### 2. Run Locally

```bash
python App/app.py --eval          # watch the trained agents play (no training, nothing saved)
python App/app.py                 # continue training from the saved checkpoints, with a window
python App/app.py --fresh         # train both agents from scratch
python App/app.py --no-render --steps 100000   # headless, full-speed training
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--eval` | off | Play with the saved weights; no learning, no saving |
| `--eval-eps` | 0.05 | Exploration rate used in `--eval` mode |
| `--fresh` | off | Ignore checkpoints and train from scratch with decaying ε |
| `--steps` | 30000 | Number of environment steps in the session |
| `--no-render` | off | No window; runs as fast as possible |
| `--fps` | 100 | Frame-rate cap when rendering |
| `--seed` | 42 | Seed for `random`, NumPy and PyTorch |

* The included checkpoints were trained from scratch on CPU (about 20 minutes) with:

  ```bash
  python App/app.py --fresh --no-render --steps 300000 --seed 42
  ```

* When training, both agents' weights are saved back to `App/agent1_dqn.pth` / `App/agent2_dqn.pth` at the end of the session, regardless of which directory you run the script from.
* Closing the window stops the session early and still saves the weights.

### 3. Run with Docker

```bash
docker-compose up --build
```

This builds a slim image with the CPU-only build of PyTorch and runs `python app.py --no-render`, which continues training headlessly at full speed. The compose file mounts `./App` into the container, so checkpoints saved during the run persist back to your local `App/` folder.

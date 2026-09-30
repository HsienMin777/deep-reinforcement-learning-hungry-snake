"""Entry point: train or watch the Gatherer (S1) vs. Hunter (S2) DQN agents.

Examples:
    python app.py                      # 接續訓練，並顯示畫面
    python app.py --fresh              # 從頭訓練
    python app.py --eval               # 只觀看訓練好的 agent (保留 5% 隨機性)，不訓練也不存檔
    python app.py --no-render --steps 100000   # 不顯示畫面，全速訓練
"""
import argparse
import os
import random

import numpy as np
import torch

from agent import Agent
from config import (DEFAULT_FPS, DEFAULT_SEED, DEFAULT_STEPS, EPS_DECAY_FRACTION,
                    EPS_END, EPS_FINETUNE, EPS_START, TARGET_UPDATE_FREQ,
                    TRAIN_EVERY, TRAIN_START)
from env import MultiSnakeGameAI

# 權重檔固定放在 app.py 旁邊，不受執行時工作目錄影響
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = {
    1: os.path.join(SCRIPT_DIR, "agent1_dqn.pth"),
    2: os.path.join(SCRIPT_DIR, "agent2_dqn.pth"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent Snake DQN: Gatherer vs. Hunter")
    parser.add_argument("--eval", action="store_true",
                        help="watch the saved agents play, no training or saving")
    parser.add_argument("--eval-eps", type=float, default=EPS_FINETUNE,
                        help=f"epsilon used in --eval mode (default: {EPS_FINETUNE}); "
                             "0 is fully greedy but tends to lock both agents into a repeating loop")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore saved checkpoints and train from scratch with decaying epsilon")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"number of environment steps in this session (default: {DEFAULT_STEPS})")
    parser.add_argument("--no-render", action="store_true",
                        help="run headless at full speed (no window)")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help=f"frame rate cap when rendering (default: {DEFAULT_FPS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed for reproducibility (default: {DEFAULT_SEED})")
    args = parser.parse_args()
    if args.eval and args.fresh:
        parser.error("--eval and --fresh cannot be used together")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_checkpoints(agents):
    for snake_id, agent in agents.items():
        path = CHECKPOINTS[snake_id]
        if not os.path.exists(path):
            print(f"Agent {snake_id}: no checkpoint at {path}, starting from scratch")
            continue
        try:
            agent.load(path)
            print(f"Agent {snake_id}: loaded {path}")
        except (RuntimeError, OSError) as e:
            print(f"Agent {snake_id}: failed to load {path} ({e}), starting from scratch")


def epsilon_at(step, total_steps, fresh):
    """從頭訓練：epsilon 在前 EPS_DECAY_FRACTION 的步數內線性下降；接續訓練：固定低值。"""
    if not fresh:
        return EPS_FINETUNE
    progress = min(1.0, step / (total_steps * EPS_DECAY_FRACTION))
    return EPS_START - (EPS_START - EPS_END) * progress


def run(args):
    set_seed(args.seed)
    training = not args.eval

    env = MultiSnakeGameAI(render=not args.no_render, fps=args.fps, max_steps=args.steps)
    agents = {1: Agent(), 2: Agent()}

    if not args.fresh:
        load_checkpoints(agents)

    mode = "Evaluation" if args.eval else ("New training" if args.fresh else "Continued training")
    print(f"Mode: {mode} | Steps: {args.steps} | Seed: {args.seed}")

    state1, state2 = env.reset_all()

    for step in range(1, args.steps + 1):
        eps = args.eval_eps if args.eval else epsilon_at(step, args.steps, args.fresh)

        action1 = agents[1].select_action(state1, eps)
        action2 = agents[2].select_action(state2, eps)

        next_state1, next_state2, reward1, reward2, dead1, dead2 = env.play_step(action1, action2)

        if training:
            # 死亡的那一步視為該 agent 的 terminal transition
            agents[1].memory.push(state1, action1, reward1, next_state1, float(dead1))
            agents[2].memory.push(state2, action2, reward2, next_state2, float(dead2))

            if step > TRAIN_START and step % TRAIN_EVERY == 0:
                agents[1].optimize()
                agents[2].optimize()

            if step % TARGET_UPDATE_FREQ == 0:
                agents[1].sync_target()
                agents[2].sync_target()

        state1, state2 = next_state1, next_state2

        if step % 1000 == 0:
            print(f"Step: {step}/{args.steps} | Score: {env.score1}-{env.score2} | Eps: {eps:.2f}")

        if env.quit_requested:
            print("Window closed, stopping early.")
            break

    print(f"Final score  S1 (food): {env.score1} | S2 (blocks): {env.score2}")

    if training:
        for snake_id, agent in agents.items():
            agent.save(CHECKPOINTS[snake_id])
        print("Models saved.")

    env.close()


if __name__ == "__main__":
    run(parse_args())

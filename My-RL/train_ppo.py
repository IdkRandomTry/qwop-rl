"""PPO components and utilities for the QWOP environment.

This module provides reusable PPO classes and functions that can be imported
into notebooks or scripts for training and evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutBuffer:
    def __init__(self, capacity: int, state_dim: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.state_dim = state_dim
        self.reset()

    def reset(self) -> None:
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action, log_prob, value, reward, done) -> None:
        self.states.append(state.copy())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.states)

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        returns = []
        advantages = []
        gae = 0.0
        values = self.values + [last_value]

        for step in reversed(range(len(self.rewards))):
            mask = 1.0 - float(self.dones[step])
            delta = (
                self.rewards[step]
                + gamma * values[step + 1] * mask
                - values[step]
            )
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def get_batches(self, batch_size: int) -> Iterator[Batch]:
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)

        states = torch.as_tensor(self.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(
            self.log_probs, dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(self.returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(
            self.advantages, dtype=torch.float32, device=self.device
        )

        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield Batch(
                states[batch_idx],
                actions[batch_idx],
                old_log_probs[batch_idx],
                returns[batch_idx],
                advantages[batch_idx],
            )

    def finalize(self, returns, advantages):
        self.returns = returns
        self.advantages = advantages


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def act(self, x):
        logits, value = self.forward(x)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, x, actions):
        logits, value = self.forward(x)
        distribution = torch.distributions.Categorical(logits=logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy, value.squeeze(-1)


def create_qwop_env(
    browser_path: str,
    driver_path: str,
    stat_in_browser: bool = True,
    game_in_browser: bool = True,
    auto_draw: bool = True,
    frames_per_step: int = 4,
    max_episode_steps: int = 5000,
    text_in_browser: str = "Training PPO Agent"
) -> gym.Env:
    """Create a QWOP environment with the specified parameters.
    
    Args:
        browser_path: Path to the browser executable
        driver_path: Path to the ChromeDriver executable
        stat_in_browser: Show statistics in browser
        game_in_browser: Show the game in browser
        auto_draw: Automatically render each frame
        frames_per_step: Number of frames per step
        max_episode_steps: Maximum steps per episode
        text_in_browser: Text to display in browser
        
    Returns:
        Configured QWOP Gymnasium environment
    """
    env = gym.make(
        'QWOP-v1',
        browser=browser_path,
        driver=driver_path,
        stat_in_browser=stat_in_browser,
        game_in_browser=game_in_browser,
        auto_draw=auto_draw,
        frames_per_step=frames_per_step,
        max_episode_steps=max_episode_steps,
        text_in_browser=text_in_browser
    )
    return env


def save_checkpoint(
    policy: ActorCritic,
    step: int,
    model_dir: str | Path,
    filename: str
) -> Path:
    """Save a model checkpoint.
    
    Args:
        policy: The policy network to save
        step: Current training step
        model_dir: Directory to save the checkpoint
        filename: Name of the checkpoint file
        
    Returns:
        Path to the saved checkpoint
    """
    ckpt_path = Path(model_dir) / filename
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "state_dict": policy.state_dict(),
            "timestamp": time.time(),
        },
        ckpt_path,
    )
    return ckpt_path


def load_checkpoint(checkpoint_path: str | Path, policy: ActorCritic, device: torch.device) -> dict:
    """Load a model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        policy: The policy network to load weights into
        device: Device to load the model on
        
    Returns:
        Dictionary containing checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['state_dict'])
    return checkpoint


def train_ppo(
    env: gym.Env,
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    total_steps: int = 200000,
    rollout_steps: int = 2048,
    batch_size: int = 256,
    ppo_epochs: int = 4,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    model_dir: str = "models",
    save_every: int = 50000,
    verbose: bool = True
) -> dict:
    """Train a PPO agent on the QWOP environment.
    
    Args:
        env: QWOP Gymnasium environment
        policy: Actor-Critic policy network
        optimizer: PyTorch optimizer for the policy
        device: Device to train on (cpu or cuda)
        total_steps: Total number of environment steps to train for
        rollout_steps: Number of steps per rollout
        batch_size: Mini-batch size for PPO updates
        ppo_epochs: Number of epochs per PPO update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_coef: PPO clipping coefficient
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm for clipping
        model_dir: Directory to save checkpoints
        save_every: Save checkpoint every N steps
        verbose: Print training progress
        
    Returns:
        Dictionary containing training statistics and final checkpoint path
    """
    state_dim = env.observation_space.shape[0]
    buffer = RolloutBuffer(rollout_steps, state_dim, device)
    
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs
    obs = obs.astype(np.float32)

    global_step = 0
    episode_reward = 0.0
    episode_len = 0
    completed_episodes = 0
    episode_rewards = []
    episode_lengths = []

    interrupted = False
    try:
        while global_step < total_steps:
            buffer.reset()
            
            # Collect rollout
            for _ in range(rollout_steps):
                global_step += 1
                tensor_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action, log_prob, value = policy.act(tensor_obs.unsqueeze(0))
                action_int = int(action.item())
                log_prob = float(log_prob.item())
                value = float(value.item())

                # Handle both old and new gym API
                step_result = env.step(action_int)
                if len(step_result) == 4:
                    next_obs, reward, done, _ = step_result
                    terminated = done
                    truncated = False
                else:
                    next_obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated

                buffer.add(obs, action_int, log_prob, value, reward, done)

                obs = next_obs.astype(np.float32)
                episode_reward += reward
                episode_len += 1

                if done:
                    if verbose:
                        print(
                            f"episode {completed_episodes + 1}: reward={episode_reward:6.2f} len={episode_len:4d}"
                        )
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_len)
                    completed_episodes += 1
                    
                    obs = env.reset()
                    if isinstance(obs, tuple):
                        obs, _ = obs
                    obs = obs.astype(np.float32)
                    episode_reward = 0.0
                    episode_len = 0

                if global_step >= total_steps:
                    break

            # Compute returns and advantages
            with torch.no_grad():
                tensor_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
                _, _, last_value = policy.act(tensor_obs.unsqueeze(0))
                last_value = float(last_value.item())

            returns, advantages = buffer.compute_returns_and_advantages(
                last_value, gamma, gae_lambda
            )
            buffer.finalize(returns, advantages)

            # PPO update
            for _ in range(ppo_epochs):
                for batch in buffer.get_batches(batch_size):
                    log_prob, entropy, values = policy.evaluate_actions(
                        batch.states, batch.actions
                    )

                    ratio = (log_prob - batch.old_log_probs).exp()
                    surr1 = ratio * batch.advantages
                    surr2 = (
                        torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                        * batch.advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = 0.5 * (batch.returns - values).pow(2).mean()
                    entropy_loss = entropy.mean()

                    loss = (
                        policy_loss
                        + value_coef * value_loss
                        - entropy_coef * entropy_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                    optimizer.step()

            # Save periodic checkpoint
            if global_step % save_every < rollout_steps:
                ckpt_path = save_checkpoint(policy, global_step, model_dir, f"ppo_step{global_step}.pt")
                if verbose:
                    print(f"[checkpoint] saved -> {ckpt_path}")

    except KeyboardInterrupt:
        interrupted = True
        if verbose:
            print("\n[info] KeyboardInterrupt received; saving final checkpoint...")
    finally:
        final_path = save_checkpoint(policy, global_step, model_dir, "ppo_final.pt")
        if verbose:
            if interrupted:
                print(f"Training interrupted. Latest policy saved to {final_path}")
            else:
                print(f"Training complete. Final policy saved to {final_path}")

    return {
        "final_checkpoint": final_path,
        "total_steps": global_step,
        "completed_episodes": completed_episodes,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "average_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
        "average_length": np.mean(episode_lengths) if episode_lengths else 0.0,
    }


def evaluate_policy(
    env: gym.Env,
    policy: ActorCritic,
    device: torch.device,
    num_episodes: int = 5,
    verbose: bool = True,
    render_delay: float = 0.02
) -> dict:
    """Evaluate a trained policy.
    
    Args:
        env: QWOP Gymnasium environment
        policy: Trained Actor-Critic policy
        device: Device to run evaluation on
        num_episodes: Number of episodes to evaluate
        verbose: Print evaluation progress
        render_delay: Delay between steps for visualization
        
    Returns:
        Dictionary containing evaluation statistics
    """
    policy.eval()
    episode_rewards = []
    episode_lengths = []
    episode_distances = []
    
    with torch.no_grad():
        for episode in range(num_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs, info = obs
            obs = obs.astype(np.float32)
            
            episode_reward = 0
            steps = 0
            
            if verbose:
                print(f"Episode {episode + 1}:")
            
            while True:
                tensor_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action, _, _ = policy.act(tensor_obs.unsqueeze(0))
                action_int = int(action.item())
                
                step_result = env.step(action_int)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                    
                obs = obs.astype(np.float32)
                episode_reward += reward
                steps += 1
                
                if done:
                    distance = info.get('distance', 0)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(steps)
                    episode_distances.append(distance)
                    
                    if verbose:
                        print(f"  Steps: {steps}")
                        print(f"  Distance: {distance:.2f}m")
                        print(f"  Total Reward: {episode_reward:.2f}")
                        print(f"  {'✓ SUCCESS!' if info.get('is_success') else '✗ Failed'}\n")
                    break
                
                if render_delay > 0:
                    time.sleep(render_delay)
    
    policy.train()
    
    results = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_distances": episode_distances,
        "average_reward": np.mean(episode_rewards),
        "average_length": np.mean(episode_lengths),
        "average_distance": np.mean(episode_distances),
        "best_distance": max(episode_distances),
        "success_rate": sum(1 for d in episode_distances if d >= 100) / num_episodes,
    }
    
    if verbose:
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Average Distance: {results['average_distance']:.2f}m")
        print(f"Best Distance: {results['best_distance']:.2f}m")
        print(f"Average Reward: {results['average_reward']:.2f}")
        print(f"Success Rate: {results['success_rate']*100:.1f}%")
        print("="*50)
    
    return results

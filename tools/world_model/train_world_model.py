from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


OBS_SIZE = 31
ACTION_COUNT = 4
ACTIONS = ((0, 1), (0, -1), (-1, 0), (1, 0))
WIDTH = 8
HEIGHT = 8
EXIT_CELL = (WIDTH - 2, HEIGHT - 2)
START_CELL = (1, 1)
MAX_FOOD = 40


@dataclass
class DatasetBundle:
    dataset: TensorDataset
    source: str
    transition_count: int
    success_rate: float


class DynamicsRewardModel(nn.Module):
    def __init__(self, obs_size: int, action_count: int, hidden_size: int) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.action_count = action_count
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size
        self.backbone = nn.Sequential(
            nn.Linear(obs_size + action_count, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.next_head = nn.Linear(hidden_size, obs_size)
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_one_hot = F.one_hot(actions, num_classes=self.action_count).float()
        x = torch.cat([obs, action_one_hot], dim=-1)
        hidden = self.backbone(x)
        return self.next_head(hidden), self.reward_head(hidden).squeeze(-1)


def load_jsonl(path: Path) -> DatasetBundle:
    observations: list[list[float]] = []
    actions: list[int] = []
    next_observations: list[list[float]] = []
    rewards: list[float] = []
    done_count = 0
    success_count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            obs = row["obs"]
            next_obs = row["next_obs"]
            if len(obs) != OBS_SIZE or len(next_obs) != OBS_SIZE:
                raise ValueError(f"line {line_number}: expected {OBS_SIZE}-value observations")
            observations.append(obs)
            actions.append(int(row["action"]))
            next_observations.append(next_obs)
            reward = float(row["reward"])
            rewards.append(reward)
            done = bool(row.get("done", False))
            outcome = str(row.get("outcome", ""))
            if done:
                done_count += 1
                if outcome == "exit" or reward > 0.5:
                    success_count += 1

    if len(observations) < 16:
        raise ValueError("need at least 16 transitions to train a useful predictor")

    dataset = TensorDataset(
        torch.tensor(observations, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(next_observations, dtype=torch.float32),
        torch.tensor(rewards, dtype=torch.float32),
    )
    return DatasetBundle(
        dataset=dataset,
        source="unity_jsonl",
        transition_count=len(observations),
        success_rate=success_count / done_count if done_count else 0.0,
    )


def synthetic_dataset(count: int = 512) -> DatasetBundle:
    observations = torch.rand(count, OBS_SIZE)
    actions = torch.randint(0, ACTION_COUNT, (count,))
    next_observations = observations.clone()
    rewards = torch.zeros(count)

    for i in range(count):
        action = int(actions[i])
        axis = 0 if action in (2, 3) else 1
        sign = -1 if action in (1, 2) else 1
        next_observations[i, axis] = torch.clamp(next_observations[i, axis] + 0.05 * sign, 0.0, 1.0)
        rewards[i] = 0.1 if action in (0, 3) else -0.05

    dataset = TensorDataset(observations, actions, next_observations, rewards)
    return DatasetBundle(
        dataset=dataset,
        source="smoke",
        transition_count=count,
        success_rate=float((rewards > 0).float().mean().item()),
    )


def synthetic_game_dataset(state_count: int = 4096, seed: int = 7) -> DatasetBundle:
    rng = random.Random(seed)
    rows: list[tuple[list[float], int, list[float], float]] = []

    for _ in range(state_count):
        state = random_state(rng)
        for action in range(ACTION_COUNT):
            rows.append(simulate_transition(state, action, rng))

    rng.shuffle(rows)
    observations, actions, next_observations, rewards = zip(*rows)
    dataset = TensorDataset(
        torch.tensor(observations, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(next_observations, dtype=torch.float32),
        torch.tensor(rewards, dtype=torch.float32),
    )
    positive_rewards = sum(1 for reward in rewards if reward > 0)
    return DatasetBundle(
        dataset=dataset,
        source="synthetic_game",
        transition_count=len(rows),
        success_rate=positive_rewards / len(rows),
    )


def random_state(rng: random.Random) -> dict:
    cells = [
        (x, y)
        for x in range(1, WIDTH - 1)
        for y in range(1, HEIGHT - 1)
        if (x, y) not in (START_CELL, EXIT_CELL)
    ]
    player = rng.choice(cells + [START_CELL])
    available = [cell for cell in cells if cell != player]
    rng.shuffle(available)

    objects = {EXIT_CELL: 1}
    for cell in available[: rng.randint(1, 3)]:
        objects[cell] = 3
    for cell in available[3 : 3 + rng.randint(1, 3)]:
        objects[cell] = 4
    for cell in available[6 : 6 + rng.randint(1, 2)]:
        objects[cell] = 2

    return {
        "player": player,
        "objects": objects,
        "food": rng.randint(12, MAX_FOOD),
        "level": rng.randint(1, 5),
    }


def simulate_transition(state: dict, action: int, rng: random.Random) -> tuple[list[float], int, list[float], float]:
    previous_obs = build_observation(state)
    previous_level = state["level"]
    previous_food = state["food"]
    previous_distance = distance_to_exit(state["player"])

    next_state = clone_state(state)
    next_state["food"] -= 1
    accepted = apply_player_action(next_state, action, rng)
    reached_exit = next_state["level"] > previous_level
    current_distance = 0 if reached_exit else distance_to_exit(next_state["player"])

    reward = -0.001
    if not accepted:
        reward -= 0.025
    reward += (previous_distance - current_distance) * 0.05
    reward += (next_state["food"] - previous_food) * 0.01

    game_over = next_state["food"] <= 0
    if reached_exit:
        reward += 1.0
    elif game_over:
        reward -= 1.0

    next_obs = build_observation(next_state)
    return previous_obs, action, next_obs, reward


def apply_player_action(state: dict, action: int, rng: random.Random) -> bool:
    dx, dy = ACTIONS[action]
    px, py = state["player"]
    target = (px + dx, py + dy)

    if not is_passable(target):
        return False

    object_code = state["objects"].get(target, 0)
    if object_code == 1:
        state.update(random_state(rng))
        state["level"] += 1
    elif object_code == 4:
        state["player"] = target
        state["food"] += 10
        del state["objects"][target]
    elif object_code in (2, 3):
        pass
    else:
        state["player"] = target

    return True


def build_observation(state: dict) -> list[float]:
    player = state["player"]
    exit_cell = EXIT_CELL
    width_scale = max(1, WIDTH - 1)
    height_scale = max(1, HEIGHT - 1)
    max_distance = max(1, WIDTH + HEIGHT)
    current_distance = distance_to_exit(player)

    values = [
        player[0] / width_scale,
        player[1] / height_scale,
        (exit_cell[0] - player[0]) / width_scale,
        (exit_cell[1] - player[1]) / height_scale,
        current_distance / max_distance,
        max(0.0, min(1.0, state["food"] / MAX_FOOD)),
        1.0 if state["food"] <= 0 else 0.0,
        max(0.0, min(1.0, state["level"] / 10.0)),
    ]

    for dx, dy in ACTIONS:
        target = (player[0] + dx, player[1] + dy)
        object_code = object_code_at(state, target)
        target_distance = abs(exit_cell[0] - target[0]) + abs(exit_cell[1] - target[1])
        values.extend(
            [
                1.0 if is_passable(target) else 0.0,
                object_code / 5.0,
                1.0 if target_distance < current_distance else 0.0,
                1.0 if object_code == 2 else 0.0,
            ]
        )

    enemy = nearest_enemy(state)
    if enemy is None:
        values.extend([0.0, 0.0, 1.0])
    else:
        enemy_cell, enemy_distance = enemy
        values.extend(
            [
                (enemy_cell[0] - player[0]) / width_scale,
                (enemy_cell[1] - player[1]) / height_scale,
                enemy_distance / max_distance,
            ]
        )

    for dx, dy in ACTIONS:
        target = (player[0] + dx, player[1] + dy)
        target_distance = abs(exit_cell[0] - target[0]) + abs(exit_cell[1] - target[1])
        values.append((current_distance - target_distance) / max_distance)

    return values[:OBS_SIZE] + [0.0] * max(0, OBS_SIZE - len(values))


def clone_state(state: dict) -> dict:
    return {
        "player": state["player"],
        "objects": dict(state["objects"]),
        "food": state["food"],
        "level": state["level"],
    }


def is_passable(cell: tuple[int, int]) -> bool:
    x, y = cell
    return 0 < x < WIDTH - 1 and 0 < y < HEIGHT - 1


def object_code_at(state: dict, cell: tuple[int, int]) -> int:
    if not is_passable(cell):
        return -1
    return state["objects"].get(cell, 0)


def distance_to_exit(cell: tuple[int, int]) -> int:
    return abs(EXIT_CELL[0] - cell[0]) + abs(EXIT_CELL[1] - cell[1])


def nearest_enemy(state: dict) -> tuple[tuple[int, int], int] | None:
    player = state["player"]
    enemies = [cell for cell, code in state["objects"].items() if code == 2]
    if not enemies:
        return None
    enemy = min(enemies, key=lambda cell: abs(player[0] - cell[0]) + abs(player[1] - cell[1]))
    return enemy, abs(player[0] - enemy[0]) + abs(player[1] - enemy[1])


def train(
    bundle: DatasetBundle,
    output_path: Path,
    metrics_path: Path,
    metrics_json_path: Path,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    reward_loss_weight: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    random.seed(seed)

    dataset = bundle.dataset
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = DynamicsRewardModel(OBS_SIZE, ACTION_COUNT, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()

    metrics: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_dynamics_loss = 0.0
        train_reward_loss = 0.0
        for obs, actions, next_obs, rewards in train_loader:
            pred_next, pred_reward = model(obs, actions)
            dynamics_loss = mse(pred_next, next_obs)
            reward_loss = mse(pred_reward, rewards)
            loss = dynamics_loss + reward_loss_weight * reward_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * obs.shape[0]
            train_dynamics_loss += dynamics_loss.item() * obs.shape[0]
            train_reward_loss += reward_loss.item() * obs.shape[0]

        train_loss /= train_size
        train_dynamics_loss /= train_size
        train_reward_loss /= train_size
        val_metrics = evaluate(model, val_loader, mse, reward_loss_weight)
        metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dynamics_loss": train_dynamics_loss,
                "train_reward_loss": train_reward_loss,
                "val_loss": val_metrics["loss"],
                "val_dynamics_loss": val_metrics["dynamics_loss"],
                "val_reward_loss": val_metrics["reward_loss"],
            }
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_dyn={val_metrics['dynamics_loss']:.6f} "
            f"val_rew={val_metrics['reward_loss']:.6f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    export_weights(model, output_path)
    write_metrics(metrics, metrics_path)
    write_summary_metrics(bundle, metrics[-1], epochs, metrics_json_path)


@torch.no_grad()
def evaluate(model: DynamicsRewardModel, loader: DataLoader, mse: nn.MSELoss, reward_loss_weight: float) -> dict[str, float]:
    model.eval()
    total = 0.0
    dynamics_total = 0.0
    reward_total = 0.0
    count = 0
    for obs, actions, next_obs, rewards in loader:
        pred_next, pred_reward = model(obs, actions)
        dynamics_loss = mse(pred_next, next_obs)
        reward_loss = mse(pred_reward, rewards)
        loss = dynamics_loss + reward_loss_weight * reward_loss
        total += loss.item() * obs.shape[0]
        dynamics_total += dynamics_loss.item() * obs.shape[0]
        reward_total += reward_loss.item() * obs.shape[0]
        count += obs.shape[0]
    count = max(1, count)
    return {
        "loss": total / count,
        "dynamics_loss": dynamics_total / count,
        "reward_loss": reward_total / count,
    }


@torch.no_grad()
def imagine(model: DynamicsRewardModel, obs: torch.Tensor, action_sequence: list[int], discount: float = 0.85) -> float:
    model.eval()
    total = 0.0
    current = obs.clone()
    if current.dim() == 1:
        current = current.unsqueeze(0)
    for step, action in enumerate(action_sequence):
        action_tensor = torch.tensor([action], dtype=torch.long, device=current.device)
        next_obs, reward = model(current, action_tensor)
        total += (discount ** step) * float(reward.item())
        current = next_obs
    return total


def export_weights(model: DynamicsRewardModel, output_path: Path) -> None:
    linear1 = model.backbone[0]
    linear2 = model.backbone[2]
    payload = {
        "schema": "rogue.world_model_weights.v2",
        "obs_size": model.obs_size,
        "action_count": model.action_count,
        "hidden_size": model.hidden_size,
        "hidden_size2": model.hidden_size2,
        "w1": linear1.weight.detach().flatten().tolist(),
        "b1": linear1.bias.detach().flatten().tolist(),
        "w2": linear2.weight.detach().flatten().tolist(),
        "b2": linear2.bias.detach().flatten().tolist(),
        "w_next": model.next_head.weight.detach().flatten().tolist(),
        "b_next": model.next_head.bias.detach().flatten().tolist(),
        "w_reward": model.reward_head.weight.detach().flatten().tolist(),
        "b_reward": model.reward_head.bias.detach().flatten().tolist(),
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"saved weights: {output_path}")


def write_metrics(metrics: list[dict[str, float]], metrics_path: Path) -> None:
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_dynamics_loss",
                "train_reward_loss",
                "val_loss",
                "val_dynamics_loss",
                "val_reward_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics)
    print(f"saved metrics: {metrics_path}")


def write_summary_metrics(bundle: DatasetBundle, final_metrics: dict[str, float], epochs: int, metrics_json_path: Path) -> None:
    payload = {
        "schema": "rogue.metrics.v1",
        "source": bundle.source,
        "transitionCount": bundle.transition_count,
        "epochs": epochs,
        "dynamicsLoss": final_metrics["val_dynamics_loss"],
        "rewardLoss": final_metrics["val_reward_loss"],
        "valLoss": final_metrics["val_loss"],
        "successRate": bundle.success_rate,
    }
    metrics_json_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"saved metrics json: {metrics_json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny world-model predictor for the Rogue Unity demo.")
    parser.add_argument("--input", type=Path, help="JSONL trajectory file exported by RogueTransitionRecorder.")
    parser.add_argument("--output", type=Path, default=Path("Assets/StreamingAssets/world_model_weights.json"))
    parser.add_argument("--metrics", type=Path, default=Path("results/world_model_metrics.csv"))
    parser.add_argument("--metrics-json", type=Path, default=Path("Assets/StreamingAssets/world_model_metrics.json"))
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--reward-loss-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke", action="store_true", help="Train on a synthetic dataset to verify the pipeline.")
    parser.add_argument("--synthetic-game", action="store_true", help="Train on generated Rogue-like one-step transitions.")
    parser.add_argument("--synthetic-game-states", type=int, default=4096)
    args = parser.parse_args()

    if args.smoke:
        bundle = synthetic_dataset()
    elif args.synthetic_game:
        bundle = synthetic_game_dataset(args.synthetic_game_states, args.seed)
    elif args.input:
        bundle = load_jsonl(args.input)
    else:
        raise SystemExit("provide --input PATH or use --smoke")

    train(
        bundle=bundle,
        output_path=args.output,
        metrics_path=args.metrics,
        metrics_json_path=args.metrics_json,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reward_loss_weight=args.reward_loss_weight,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

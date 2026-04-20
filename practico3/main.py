import gymnasium
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from ucs_agent import UCSAgent
from a_star_agent import AStarAgent
from model import Model
import time
import traceback

envs = [
    (
        "4x4",
        gymnasium.make(
            "FrozenLake-v1",
            desc=generate_random_map(size=4),
            is_slippery=False,
            render_mode="rgb_array",
        ),
    ),
    (
        "8x8",
        gymnasium.make(
            "FrozenLake-v1",
            desc=generate_random_map(size=8),
            is_slippery=False,
            render_mode="rgb_array",
        ),
    ),
    (
        "16x16",
        gymnasium.make(
            "FrozenLake-v1",
            desc=generate_random_map(size=16),
            is_slippery=False,
            render_mode="rgb_array",
        ),
    ),
    (
        "32x32",
        gymnasium.make(
            "FrozenLake-v1",
            desc=generate_random_map(size=32),
            is_slippery=False,
            render_mode="rgb_array",
        ),
    ),
]

agents = [
    ("UCS", UCSAgent, {}),
    ("A* Manhattan", AStarAgent, {"heuristic_type": "manhattan"}),
    ("A* Euclidean", AStarAgent, {"heuristic_type": "euclidean"}),
]


def find_states(desc):
    nrows, ncols = desc.shape
    initial_state = None
    end_state = None
    for row in range(nrows):
        for col in range(ncols):
            cell = desc[row, col].decode('utf-8')
            if cell == 'S':
                initial_state = row * ncols + col
            elif cell == 'G':
                end_state = row * ncols + col
    return initial_state, end_state


def main():
    header = f"{'Agente':<20} {'Ambiente':<10} {'Reward':<10} {'Pasos':<10} {'Tiempo (s)':<12}"
    print(header)
    print("-" * len(header))

    for env_name, env in envs:
        desc = env.unwrapped.desc
        model = Model(desc)
        initial_state, end_state = find_states(desc)

        for agent_name, AgentClass, kwargs in agents:
            try:
                agent = AgentClass(env, initial_state, end_state, model, **kwargs)
                start = time.time()
                reward, steps = agent.run()
                elapsed = time.time() - start
                print(f"{agent_name:<20} {env_name:<10} {reward:<10.1f} {steps:<10} {elapsed:<12.4f}")
            except Exception:
                print(f"{agent_name:<20} {env_name:<10} {'SIN SOLUCION':<10}")
                traceback.print_exc()

        print()


if __name__ == "__main__":
    main()

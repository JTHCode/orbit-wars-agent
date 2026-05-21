"""Reproducible A/B benchmark harness and seed archetype configuration."""

# Benchmark harness for reproducible A/B testing across seed archetypes.
import time
from kaggle_environments import make

SEED_BUCKETS = {
    "low_prod__mostly_rotating__big_rotating": [647, 2177, 7314],
    "med_low_prod__mostly_static__big_static": [1990, 2393, 9427],
    "med_high_prod__mixed_static__big_rotating": [1265, 4461, 8855],
    "high_prod__mostly_rotating__big_static": [405, 9408, 9658],
}

def run_ab_benchmark(agent_a="main.py", agent_b="random", episode_steps=500):
    """Run 12 seeded games and report per-game plus per-archetype outcomes."""
    total_games = sum(len(v) for v in SEED_BUCKETS.values())
    archetype_wins = {k: 0 for k in SEED_BUCKETS}
    game_durations = []

    print(f"Starting A/B benchmark: {agent_a} (Player 0) vs {agent_b} (Player 1)")
    print(f"Total games: {total_games}")

    game_idx = 0
    for archetype, seeds in SEED_BUCKETS.items():
        for seed in seeds:
            game_idx += 1
            t0 = time.perf_counter()
            env = make(
                "orbit_wars",
                configuration={"seed": int(seed), "episodeSteps": int(episode_steps)},
                debug=False,
            )
            env.run([agent_a, agent_b])
            elapsed = time.perf_counter() - t0
            game_durations.append(elapsed)

            final = env.steps[-1]
            p0_reward = final[0].reward if final[0].reward is not None else float("-inf")
            p1_reward = final[1].reward if final[1].reward is not None else float("-inf")

            if p0_reward > p1_reward:
                winner = "agent_a"
                archetype_wins[archetype] += 1
            elif p1_reward > p0_reward:
                winner = "agent_b"
            else:
                winner = "draw"

            turns = max(0, len(env.steps) - 1)
            print(
                f"[{game_idx:02d}/{total_games}] archetype={archetype} seed={seed} "
                f"winner={winner} turns={turns} time_sec={elapsed:.3f}"
            )

    avg_time = sum(game_durations) / len(game_durations) if game_durations else 0.0

    print("\n=== Final Benchmark Summary ===")
    print(f"Agent A: {agent_a} | Agent B: {agent_b}")
    print("Agent A wins by archetype:")
    for archetype, wins in archetype_wins.items():
        print(f"  - {archetype}: {wins}/{len(SEED_BUCKETS[archetype])}")
    print(f"Total Agent A wins: {sum(archetype_wins.values())}/{total_games}")
    print(f"Overall average game time (sec): {avg_time:.3f}")

# Example usage:
# run_ab_benchmark(agent_a="main.py", agent_b="random")

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


def _bucket_turn(turn, episode_steps):
    frac = 0.0 if episode_steps <= 0 else float(turn) / float(episode_steps)
    if frac < 0.2:
        return "t00_20"
    if frac < 0.4:
        return "t20_40"
    if frac < 0.6:
        return "t40_60"
    if frac < 0.8:
        return "t60_80"
    return "t80_100"


def _dominant_midgame_phase(transitions, fallback):
    counts = {}
    for ev in transitions:
        t = ev.get("turn", 0)
        if 0.35 <= t / 500.0 <= 0.75:
            ph = ev.get("to")
            if ph:
                counts[ph] = counts.get(ph, 0) + 1
    if not counts:
        return fallback
    return max(counts, key=counts.get)

def run_ab_benchmark(agent_a="main.py", agent_b="random", episode_steps=500):
    """Run 12 seeded games and report per-game plus per-archetype outcomes."""
    total_games = sum(len(v) for v in SEED_BUCKETS.values())
    archetype_wins = {k: 0 for k in SEED_BUCKETS}
    game_durations = []
    phase_occupancy = {}
    transitions_per_game = []
    trajectory_stats = {}

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

            debug_payload = None
            for step in env.steps:
                for st in step:
                    obs = getattr(st, "observation", None)
                    if isinstance(obs, dict) and "debug" in obs:
                        debug_payload = obs.get("debug")
            phase_diag = (debug_payload or {}).get("phase_diagnostics", {})
            transitions = (debug_payload or {}).get("phase_transitions", [])
            current_phase = phase_diag.get("current_phase", "unknown")
            bucket = _bucket_turn(phase_diag.get("game_turn", 0), episode_steps)
            phase_occupancy.setdefault(bucket, {})
            phase_occupancy[bucket][current_phase] = phase_occupancy[bucket].get(current_phase, 0) + 1
            transitions_per_game.append(len(transitions))
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

            traj = _dominant_midgame_phase(transitions, current_phase)
            trajectory_stats.setdefault(traj, {"wins": 0, "games": 0})
            trajectory_stats[traj]["games"] += 1
            if winner == "agent_a":
                trajectory_stats[traj]["wins"] += 1

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

    print("Phase occupancy by turn bucket:")
    for bucket in sorted(phase_occupancy.keys()):
        print(f"  - {bucket}: {phase_occupancy[bucket]}")

    avg_transitions = (sum(transitions_per_game) / len(transitions_per_game)) if transitions_per_game else 0.0
    print(f"Phase transition count per game (avg): {avg_transitions:.2f}")

    print("Win-rate by dominant midgame phase trajectory:")
    for traj, stats in sorted(trajectory_stats.items()):
        wr = stats["wins"] / max(1, stats["games"])
        delta = wr - (sum(archetype_wins.values()) / max(1, total_games))
        print(f"  - {traj}: win_rate={wr:.3f} delta_vs_overall={delta:+.3f} games={stats['games']}")

# Example usage:
# run_ab_benchmark(agent_a="main.py", agent_b="random")

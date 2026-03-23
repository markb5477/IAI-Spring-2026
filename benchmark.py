#!/usr/bin/env python3
"""Benchmark suite for Kalaha AI agents.

Runs automated games between agent pairs, swapping first-player to
control for Kalaha's first-mover advantage, and prints a summary table.

Usage:
    python benchmark.py                  # default: 50 games per matchup
    python benchmark.py --games 100      # more games for tighter statistics
    python benchmark.py --seed 42        # reproducible random agent
"""

import argparse
import random
import time
import copy

from src.game import get_initial_board, make_move
from src.agents import make_random_agent, make_greedy_agent, make_minimax_agent
from src.heuristics import evaluate_sigmoid, evaluate_linear


# ── Game simulation ──────────────────────────────────────────────────

def simulate_game(agent_p1, agent_p2):
    """Play one full game between two agents.

    Returns:
        (p1_score, p2_score, num_turns, elapsed_seconds)
    """
    board = get_initial_board()
    turn = 1          # player 1 starts
    num_turns = 0
    t0 = time.perf_counter()

    while not (sum(board[1:7]) == 0 or sum(board[8:14]) == 0):
        if turn == 1:
            move = agent_p1(board, 1)
        else:
            move = agent_p2(board, 2)
        board, turn = make_move(board, turn, move, silent=True)
        num_turns += 1

        # safety valve against infinite loops
        if num_turns > 500:
            break

    # end-of-game: sweep remaining pits into stores
    board[7] += sum(board[1:7])
    board[14] += sum(board[8:14])
    for i in range(1, 7):
        board[i] = 0
    for i in range(8, 14):
        board[i] = 0

    elapsed = time.perf_counter() - t0
    return board[7], board[14], num_turns, elapsed


# ── Matchup runner ───────────────────────────────────────────────────

def run_matchup(agent_a, agent_b, num_games):
    """Run num_games with A as P1 and num_games with A as P2.

    Returns a dict with aggregated results from agent A's perspective.
    """
    a_wins = 0
    b_wins = 0
    draws = 0
    a_wins_as_p1 = 0
    a_wins_as_p2 = 0
    total_turns = 0
    total_time = 0.0

    total = num_games * 2  # half as P1, half as P2

    for i in range(num_games):
        # A plays as Player 1
        s1, s2, turns, elapsed = simulate_game(agent_a, agent_b)
        total_turns += turns
        total_time += elapsed
        if s1 > s2:
            a_wins += 1
            a_wins_as_p1 += 1
        elif s2 > s1:
            b_wins += 1
        else:
            draws += 1

    for i in range(num_games):
        # A plays as Player 2
        s1, s2, turns, elapsed = simulate_game(agent_b, agent_a)
        total_turns += turns
        total_time += elapsed
        if s2 > s1:
            a_wins += 1
            a_wins_as_p2 += 1
        elif s1 > s2:
            b_wins += 1
        else:
            draws += 1

    return {
        "a_name": agent_a.name,
        "b_name": agent_b.name,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "total_games": total,
        "a_win_pct": a_wins / total * 100,
        "a_wins_as_p1": a_wins_as_p1,
        "a_wins_as_p2": a_wins_as_p2,
        "avg_turns": total_turns / total,
        "avg_time": total_time / total,
    }


# ── Table printer ────────────────────────────────────────────────────

def print_results(results_list):
    header = (
        f"{'Matchup (A vs B)':<35} "
        f"{'A Wins':>6} {'B Wins':>6} {'Draws':>5} "
        f"{'A Win%':>6} "
        f"{'A as P1':>7} {'A as P2':>7} "
        f"{'Avg Turns':>9} {'Avg Time':>9}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results_list:
        label = f"{r['a_name']} vs {r['b_name']}"
        print(
            f"{label:<35} "
            f"{r['a_wins']:>6} {r['b_wins']:>6} {r['draws']:>5} "
            f"{r['a_win_pct']:>5.1f}% "
            f"{r['a_wins_as_p1']:>7} {r['a_wins_as_p2']:>7} "
            f"{r['avg_turns']:>9.1f} {r['avg_time']:>8.3f}s"
        )
    print(sep)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Kalaha AI agents")
    parser.add_argument("--games", type=int, default=50,
                        help="Games per side per matchup (total = 2x this). Default: 50")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # ── Define agents ────────────────────────────────────────────────
    agents = {
        "random":     make_random_agent(),
        "greedy":     make_greedy_agent(),
        "linear_d3":  make_minimax_agent(3, evaluate_linear,  "Linear-d3"),
        "linear_d5":  make_minimax_agent(5, evaluate_linear,  "Linear-d5"),
        "sigmoid_d3": make_minimax_agent(3, evaluate_sigmoid, "Sigmoid-d3"),
        "sigmoid_d5": make_minimax_agent(5, evaluate_sigmoid, "Sigmoid-d5"),
        "sigmoid_d8": make_minimax_agent(8, evaluate_sigmoid, "Sigmoid-d8"),
    }

    # ── Define matchups (A vs B) ─────────────────────────────────────
    matchups = [
        # Baselines: sigmoid-d5 vs weak opponents
        ("sigmoid_d5", "random"),
        ("sigmoid_d5", "greedy"),

        # Heuristic comparison at same depth
        ("sigmoid_d3", "linear_d3"),
        ("sigmoid_d5", "linear_d5"),

        # Depth scaling (same heuristic)
        ("sigmoid_d5", "sigmoid_d3"),
        ("sigmoid_d8", "sigmoid_d5"),

        # Linear depth scaling
        ("linear_d5",  "linear_d3"),
    ]

    # ── Run benchmarks ───────────────────────────────────────────────
    print(f"Running benchmarks: {args.games} games per side ({args.games * 2} total per matchup)")
    print(f"Seed: {args.seed}\n")

    results = []
    for a_key, b_key in matchups:
        agent_a, agent_b = agents[a_key], agents[b_key]
        label = f"  {agent_a.name} vs {agent_b.name}..."
        print(label, end=" ", flush=True)
        r = run_matchup(agent_a, agent_b, args.games)
        print(f"done ({r['avg_time']:.3f}s/game avg)")
        results.append(r)

    print_results(results)


if __name__ == "__main__":
    main()

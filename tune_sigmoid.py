#!/usr/bin/env python3
"""Tune sigmoid heuristic parameters to find configurations that beat linear."""

import math
import itertools
import time

from src.ai_logic import minimax
from src.game import get_initial_board, make_move
from src.heuristics import evaluate_linear
from src.agents import make_minimax_agent


def make_sigmoid_heuristic(steepness, center, nest_w, pit_w, extra_w):
    """Create a sigmoid heuristic with the given parameters."""
    def evaluate(board, ai_player):
        if ai_player == 2:
            my_nest, opp_nest = board[14], board[7]
            my_pits = sum(i * board[7 + i] for i in range(1, 7))
            opp_pits = sum(i * board[i] for i in range(1, 7))
        else:
            my_nest, opp_nest = board[7], board[14]
            my_pits = sum(i * board[i] for i in range(1, 7))
            opp_pits = sum(i * board[7 + i] for i in range(1, 7))

        def win_prob(n):
            return 1 / (1 + math.exp(-steepness * (n - center)))

        nest_score = (win_prob(my_nest) - win_prob(opp_nest)) * nest_w
        pit_score = (my_pits - opp_pits) * pit_w

        if ai_player == 1:
            extra_turn_bonus = sum(1 for i in range(1, 7) if board[i] == (7 - i))
        else:
            extra_turn_bonus = sum(1 for i in range(1, 7) if board[7 + i] == (7 - i))

        return nest_score + pit_score + extra_turn_bonus * extra_w

    return evaluate


def simulate_game(agent_p1, agent_p2):
    board = get_initial_board()
    turn = 1
    num_turns = 0
    while not (sum(board[1:7]) == 0 or sum(board[8:14]) == 0):
        if turn == 1:
            move = agent_p1(board, 1)
        else:
            move = agent_p2(board, 2)
        board, turn = make_move(board, turn, move, silent=True)
        num_turns += 1
        if num_turns > 500:
            break
    board[7] += sum(board[1:7])
    board[14] += sum(board[8:14])
    return board[7], board[14]


def test_config(steepness, center, nest_w, pit_w, extra_w, depth=5):
    """Test a sigmoid config vs linear at a given depth. Returns (sigmoid_wins, linear_wins, draws)."""
    sig_fn = make_sigmoid_heuristic(steepness, center, nest_w, pit_w, extra_w)
    sig_agent = make_minimax_agent(depth, sig_fn, "Sigmoid")
    lin_agent = make_minimax_agent(depth, evaluate_linear, "Linear")

    sig_wins, lin_wins, draws = 0, 0, 0

    # Sigmoid as P1
    s1, s2 = simulate_game(sig_agent, lin_agent)
    if s1 > s2:   sig_wins += 1
    elif s2 > s1:  lin_wins += 1
    else:          draws += 1

    # Sigmoid as P2
    s1, s2 = simulate_game(lin_agent, sig_agent)
    if s2 > s1:   sig_wins += 1
    elif s1 > s2:  lin_wins += 1
    else:          draws += 1

    return sig_wins, lin_wins, draws


def main():
    # Parameter grids to search
    steepness_vals = [0.1, 0.2, 0.35, 0.5, 0.8, 1.0]
    nest_w_vals    = [20, 50, 100, 200, 500]
    pit_w_vals     = [0.2, 0.5, 0.8, 1.5, 3.0, 5.0]
    extra_w_vals   = [0, 1, 3, 5, 10]
    center         = 24  # fixed: always half of 48

    depth = 5

    total_configs = len(steepness_vals) * len(nest_w_vals) * len(pit_w_vals) * len(extra_w_vals)
    print(f"Testing {total_configs} parameter configurations at depth {depth}")
    print(f"Each config plays 2 games vs Linear-d{depth} (one per side)\n")

    winners = []
    ties = []
    tested = 0
    t_start = time.perf_counter()

    for steepness in steepness_vals:
        for nest_w in nest_w_vals:
            for pit_w in pit_w_vals:
                for extra_w in extra_w_vals:
                    tested += 1
                    sw, lw, d = test_config(steepness, center, nest_w, pit_w, extra_w, depth)

                    tag = f"s={steepness} nw={nest_w} pw={pit_w} ew={extra_w}"

                    if sw > lw:
                        winners.append((sw, lw, d, tag))
                    elif sw == lw:
                        ties.append((sw, lw, d, tag))

                    if tested % 50 == 0:
                        elapsed = time.perf_counter() - t_start
                        eta = elapsed / tested * (total_configs - tested)
                        print(f"  [{tested}/{total_configs}] "
                              f"{len(winners)} winning, {len(ties)} tied, "
                              f"ETA {eta:.0f}s", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed:.1f}s. Tested {total_configs} configs.\n")

    # ── Results ──────────────────────────────────────────────────────
    print(f"{'='*70}")
    print(f"CONFIGS THAT BEAT LINEAR-d{depth}: {len(winners)}")
    print(f"{'='*70}")

    if winners:
        # Sort by sigmoid wins descending
        winners.sort(key=lambda x: x[0], reverse=True)
        print(f"\n{'Parameters':<45} {'Sig':>3} {'Lin':>3} {'Draw':>4}")
        print("-" * 60)
        for sw, lw, d, tag in winners:
            print(f"{tag:<45} {sw:>3} {lw:>3} {d:>4}")
    else:
        print("\nNo sigmoid configuration beat linear at this depth.")

    print(f"\n{'='*70}")
    print(f"CONFIGS THAT TIED LINEAR-d{depth}: {len(ties)}")
    print(f"{'='*70}")

    if ties:
        ties.sort(key=lambda x: x[2], reverse=True)  # sort by draws
        print(f"\n{'Parameters':<45} {'Sig':>3} {'Lin':>3} {'Draw':>4}")
        print("-" * 60)
        for sw, lw, d, tag in ties[:30]:  # show top 30
            print(f"{tag:<45} {sw:>3} {lw:>3} {d:>4}")
        if len(ties) > 30:
            print(f"  ... and {len(ties) - 30} more tied configs")


if __name__ == "__main__":
    main()

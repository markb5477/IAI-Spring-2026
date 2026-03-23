#!/usr/bin/env python3
"""Tune sigmoid vs linear at depth 10 with a small grid and concurrent games."""

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.game import get_initial_board, make_move
from src.ai_logic import minimax
from src.heuristics import evaluate_linear


def make_sigmoid_fn(steepness, nest_w, pit_w, extra_w):
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
            return 1 / (1 + math.exp(-steepness * (n - 24)))

        nest_score = (win_prob(my_nest) - win_prob(opp_nest)) * nest_w
        pit_score = (my_pits - opp_pits) * pit_w

        if ai_player == 1:
            extra = sum(1 for i in range(1, 7) if board[i] == (7 - i))
        else:
            extra = sum(1 for i in range(1, 7) if board[7 + i] == (7 - i))

        return nest_score + pit_score + extra * extra_w
    return evaluate


def simulate_game(agent_p1_fn, agent_p2_fn):
    board = get_initial_board()
    turn = 1
    turns = 0
    while not (sum(board[1:7]) == 0 or sum(board[8:14]) == 0):
        if turn == 1:
            move = agent_p1_fn(board, 1)
        else:
            move = agent_p2_fn(board, 2)
        board, turn = make_move(board, turn, move, silent=True)
        turns += 1
        if turns > 500:
            break
    board[7] += sum(board[1:7])
    board[14] += sum(board[8:14])
    return board[7], board[14]


def make_agent_fn(depth, eval_fn):
    def agent(board, player):
        _, move = minimax(board, depth, float('-inf'), float('inf'),
                          player, player, make_move, eval_fn)
        return move
    return agent


def run_one_config(args):
    """Run a single config: sigmoid as P1 then as P2 vs linear. Returns result dict."""
    steepness, nest_w, pit_w, extra_w, depth = args
    sig_fn = make_sigmoid_fn(steepness, nest_w, pit_w, extra_w)
    sig = make_agent_fn(depth, sig_fn)
    lin = make_agent_fn(depth, evaluate_linear)

    sig_wins, lin_wins = 0, 0

    s1, s2 = simulate_game(sig, lin)  # sigmoid as P1
    if s1 > s2: sig_wins += 1
    elif s2 > s1: lin_wins += 1

    s1, s2 = simulate_game(lin, sig)  # sigmoid as P2
    if s2 > s1: sig_wins += 1
    elif s1 > s2: lin_wins += 1

    tag = f"s={steepness} nw={nest_w} pw={pit_w} ew={extra_w}"
    return {"tag": tag, "sig": sig_wins, "lin": lin_wins, "draw": 2 - sig_wins - lin_wins}


def main():
    depth = 10

    # Small grid with large jumps
    configs = []
    for steepness in [0.1, 0.35, 0.8]:
        for nest_w in [50, 200, 500]:
            for pit_w in [0.2, 1.0, 5.0]:
                for extra_w in [0, 3, 10]:
                    configs.append((steepness, nest_w, pit_w, extra_w, depth))

    # Also add current defaults
    configs.append((0.35, 100, 0.8, 3, depth))

    print(f"Testing {len(configs)} configs at depth {depth} using parallel workers\n")
    t0 = time.perf_counter()

    winners, ties, losers = [], [], []

    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(run_one_config, c): c for c in configs}
        done = 0
        for future in as_completed(futures):
            done += 1
            r = future.result()
            if r["sig"] > r["lin"]:
                winners.append(r)
            elif r["sig"] == r["lin"]:
                ties.append(r)
            else:
                losers.append(r)
            print(f"  [{done}/{len(configs)}] {r['tag']:>40}  "
                  f"Sig {r['sig']} - Lin {r['lin']}  "
                  f"{'WIN' if r['sig'] > r['lin'] else 'TIE' if r['sig'] == r['lin'] else 'LOSS'}",
                  flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s\n")

    print(f"{'='*60}")
    print(f"SIGMOID WINS: {len(winners)}  |  TIES: {len(ties)}  |  LOSSES: {len(losers)}")
    print(f"{'='*60}")

    if winners:
        print(f"\n  WINNING CONFIGS:")
        print(f"  {'Parameters':<40} {'Sig':>3} {'Lin':>3}")
        print(f"  {'-'*50}")
        for r in sorted(winners, key=lambda x: x['sig'], reverse=True):
            print(f"  {r['tag']:<40} {r['sig']:>3} {r['lin']:>3}")

    if ties:
        print(f"\n  TIED CONFIGS:")
        print(f"  {'Parameters':<40} {'Sig':>3} {'Lin':>3}")
        print(f"  {'-'*50}")
        for r in ties:
            print(f"  {r['tag']:<40} {r['sig']:>3} {r['lin']:>3}")


if __name__ == "__main__":
    main()

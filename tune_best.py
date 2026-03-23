#!/usr/bin/env python3
"""Find the best linear and sigmoid configs via elimination, then finals."""

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.game import get_initial_board, make_move
from src.ai_logic import minimax


# ── Heuristics (module-level for pickling) ───────────────────────────

def eval_linear(board, ai_player, nest_w, pit_w):
    if ai_player == 2:
        nest_diff = board[14] - board[7]
        pit_diff = sum(board[8:14]) - sum(board[1:7])
    else:
        nest_diff = board[7] - board[14]
        pit_diff = sum(board[1:7]) - sum(board[8:14])
    return nest_diff * nest_w + pit_diff * pit_w


def eval_sigmoid(board, ai_player, steepness, nest_w, pit_w, extra_w):
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


# ── Wrappers that minimax can call (need 2-arg signature) ────────────

def make_linear_eval(params):
    nw, pw = params
    def fn(board, ai_player):
        return eval_linear(board, ai_player, nw, pw)
    return fn


def make_sigmoid_eval(params):
    s, nw, pw, ew = params
    def fn(board, ai_player):
        return eval_sigmoid(board, ai_player, s, nw, pw, ew)
    return fn


# ── Game simulation ──────────────────────────────────────────────────

def simulate_game_with_params(p1_type, p1_params, p2_type, p2_params, depth):
    """Simulate a full game. Types: 'lin' or 'sig'. Params are tuples."""
    if p1_type == 'lin':
        p1_fn = make_linear_eval(p1_params)
    else:
        p1_fn = make_sigmoid_eval(p1_params)

    if p2_type == 'lin':
        p2_fn = make_linear_eval(p2_params)
    else:
        p2_fn = make_sigmoid_eval(p2_params)

    board = get_initial_board()
    turn = 1
    turns = 0
    while not (sum(board[1:7]) == 0 or sum(board[8:14]) == 0):
        if turn == 1:
            _, move = minimax(board, depth, float('-inf'), float('inf'),
                              1, 1, make_move, p1_fn)
        else:
            _, move = minimax(board, depth, float('-inf'), float('inf'),
                              2, 2, make_move, p2_fn)
        board, turn = make_move(board, turn, move, silent=True)
        turns += 1
        if turns > 500:
            break
    board[7] += sum(board[1:7])
    board[14] += sum(board[8:14])
    return board[7], board[14]


# ── Worker: candidate vs baseline, both sides ───────────────────────

def worker_vs_baseline(args):
    label, cand_type, cand_params, base_type, base_params, depth = args

    cand_wins, base_wins = 0, 0
    cand_stones, base_stones = 0, 0

    # Candidate as P1
    s1, s2 = simulate_game_with_params(cand_type, cand_params, base_type, base_params, depth)
    cand_stones += s1; base_stones += s2
    if s1 > s2: cand_wins += 1
    elif s2 > s1: base_wins += 1

    # Candidate as P2
    s1, s2 = simulate_game_with_params(base_type, base_params, cand_type, cand_params, depth)
    cand_stones += s2; base_stones += s1
    if s2 > s1: cand_wins += 1
    elif s1 > s2: base_wins += 1

    return {
        "label": label, "cand_wins": cand_wins, "base_wins": base_wins,
        "cand_stones": cand_stones, "base_stones": base_stones,
        "stone_diff": cand_stones - base_stones,
    }


def worker_rr_match(args):
    a_label, a_type, a_params, b_label, b_type, b_params, depth = args

    aw, bw = 0, 0
    ast, bst = 0, 0

    # A as P1
    s1, s2 = simulate_game_with_params(a_type, a_params, b_type, b_params, depth)
    ast += s1; bst += s2
    if s1 > s2: aw += 1
    elif s2 > s1: bw += 1

    # A as P2
    s1, s2 = simulate_game_with_params(b_type, b_params, a_type, a_params, depth)
    ast += s2; bst += s1
    if s2 > s1: aw += 1
    elif s1 > s2: bw += 1

    return {"a": a_label, "b": b_label, "aw": aw, "bw": bw,
            "ast": ast, "bst": bst}


# ── Ranking and tournament functions ─────────────────────────────────

def rank_against_baseline(candidates, base_type, base_params, depth, title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"  {len(candidates)} candidates at depth {depth}")
    print(f"{'='*65}")

    tasks = []
    for label, (ctype, cparams) in candidates.items():
        tasks.append((label, ctype, cparams, base_type, base_params, depth))

    results = []
    done = 0
    t0 = time.perf_counter()

    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(worker_vs_baseline, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            results.append(future.result())
            if done % 20 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - t0
                eta = elapsed / done * (len(tasks) - done) if done else 0
                print(f"  [{done}/{len(tasks)}] ETA {eta:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s\n")

    results.sort(key=lambda x: (x["cand_wins"], x["stone_diff"]), reverse=True)

    print(f"  {'#':<3} {'Config':<42} {'W':>2} {'L':>2} {'Stones':>7} {'Diff':>6}")
    print(f"  {'-'*60}")
    for rank, r in enumerate(results[:20], 1):
        diff = f"{'+' if r['stone_diff']>=0 else ''}{r['stone_diff']}"
        print(f"  {rank:<3} {r['label']:<42} {r['cand_wins']:>2} {r['base_wins']:>2} "
              f"{r['cand_stones']:>7} {diff:>6}")

    return results


def round_robin(finalists, depth, title):
    names = list(finalists.keys())
    n = len(names)
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"  {n} finalists, {n*(n-1)//2} matchups at depth {depth}")
    print(f"{'='*70}")

    tasks = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = names[i], names[j]
            at, ap = finalists[a]
            bt, bp = finalists[b]
            tasks.append((a, at, ap, b, bt, bp, depth))

    scores = {name: {"w": 0, "l": 0, "d": 0, "sf": 0, "sa": 0} for name in names}
    done = 0
    t0 = time.perf_counter()

    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(worker_rr_match, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            r = future.result()
            a, b = r["a"], r["b"]
            dr = 2 - r["aw"] - r["bw"]
            scores[a]["w"] += r["aw"]; scores[a]["l"] += r["bw"]; scores[a]["d"] += dr
            scores[a]["sf"] += r["ast"]; scores[a]["sa"] += r["bst"]
            scores[b]["w"] += r["bw"]; scores[b]["l"] += r["aw"]; scores[b]["d"] += dr
            scores[b]["sf"] += r["bst"]; scores[b]["sa"] += r["ast"]

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s\n")

    board = []
    for name in names:
        s = scores[name]
        games = s["w"] + s["l"] + s["d"]
        sd = s["sf"] - s["sa"]
        wp = s["w"] / games * 100 if games else 0
        board.append({"name": name, "w": s["w"], "l": s["l"], "d": s["d"],
                       "games": games, "wp": wp, "sd": sd,
                       "avg": s["sf"] / games if games else 0})
    board.sort(key=lambda x: (x["w"], x["sd"]), reverse=True)

    print(f"  {'#':<3} {'Config':<42} {'W':>3} {'L':>3} {'D':>3} {'Win%':>6} {'Diff':>6} {'AvgScr':>6}")
    print(f"  {'-'*75}")
    for rank, e in enumerate(board, 1):
        diff = f"{'+' if e['sd']>=0 else ''}{e['sd']}"
        print(f"  {rank:<3} {e['name']:<42} {e['w']:>3} {e['l']:>3} "
              f"{e['d']:>3} {e['wp']:>5.1f}% {diff:>6} {e['avg']:>6.1f}")

    return board


def main():
    depth = 10

    # ── PHASE 1: Linear tuning ───────────────────────────────────────
    linear_cands = {}
    for nw in [5, 10, 15, 20, 25, 30, 40, 50]:
        for pw in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
            label = f"Lin nw={nw} pw={pw}"
            linear_cands[label] = ('lin', (nw, pw))

    lin_results = rank_against_baseline(
        linear_cands, 'lin', (20, 1), depth,
        "PHASE 1: LINEAR TUNING (48 configs vs Linear(20,1))")

    # Get best linear params
    best_lin = lin_results[0]
    best_lin_label = best_lin["label"]
    best_lin_entry = linear_cands[best_lin_label]
    print(f"\n  >> Best linear: {best_lin_label}\n")

    # ── PHASE 2: Sigmoid tuning vs best linear ──────────────────────
    sigmoid_cands = {}
    for steep in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        for nw in [50, 100, 200, 300, 500]:
            for pw in [0.2, 0.5, 1.0, 3.0, 5.0]:
                for ew in [0, 3, 5, 10]:
                    label = f"Sig s={steep} n={nw} p={pw} e={ew}"
                    sigmoid_cands[label] = ('sig', (steep, nw, pw, ew))

    sig_results = rank_against_baseline(
        sigmoid_cands, best_lin_entry[0], best_lin_entry[1], depth,
        f"PHASE 2: SIGMOID TUNING (600 configs vs {best_lin_label})")

    # ── PHASE 3: Finals ──────────────────────────────────────────────
    finalists = {}
    for r in lin_results[:5]:
        finalists[r["label"]] = linear_cands[r["label"]]
    for r in sig_results[:5]:
        finalists[r["label"]] = sigmoid_cands[r["label"]]

    round_robin(finalists, depth, "PHASE 3: FINALS (Top 5 Linear vs Top 5 Sigmoid)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Focused tuning: smaller sigmoid grid vs best linear, then finals."""

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.game import get_initial_board, make_move
from src.ai_logic import minimax


# ── Heuristics (module-level for pickling) ───────────────────────────

def eval_linear(board, ai_player, nest_w, pit_w):
    if ai_player == 2:
        return (board[14] - board[7]) * nest_w + (sum(board[8:14]) - sum(board[1:7])) * pit_w
    return (board[7] - board[14]) * nest_w + (sum(board[1:7]) - sum(board[8:14])) * pit_w


def eval_sigmoid(board, ai_player, steepness, nest_w, pit_w, extra_w):
    if ai_player == 2:
        my_nest, opp_nest = board[14], board[7]
        my_pits = sum(i * board[7 + i] for i in range(1, 7))
        opp_pits = sum(i * board[i] for i in range(1, 7))
    else:
        my_nest, opp_nest = board[7], board[14]
        my_pits = sum(i * board[i] for i in range(1, 7))
        opp_pits = sum(i * board[7 + i] for i in range(1, 7))

    wp = lambda n: 1 / (1 + math.exp(-steepness * (n - 24)))
    nest_score = (wp(my_nest) - wp(opp_nest)) * nest_w
    pit_score = (my_pits - opp_pits) * pit_w

    if ai_player == 1:
        extra = sum(1 for i in range(1, 7) if board[i] == (7 - i))
    else:
        extra = sum(1 for i in range(1, 7) if board[7 + i] == (7 - i))

    return nest_score + pit_score + extra * extra_w


def _make_eval(kind, params):
    if kind == 'lin':
        nw, pw = params
        return lambda b, p: eval_linear(b, p, nw, pw)
    else:
        s, nw, pw, ew = params
        return lambda b, p: eval_sigmoid(b, p, s, nw, pw, ew)


def simulate(p1_kind, p1_params, p2_kind, p2_params, depth):
    p1_fn = _make_eval(p1_kind, p1_params)
    p2_fn = _make_eval(p2_kind, p2_params)
    board = get_initial_board()
    turn = 1
    turns = 0
    while not (sum(board[1:7]) == 0 or sum(board[8:14]) == 0):
        fn = p1_fn if turn == 1 else p2_fn
        _, move = minimax(board, depth, float('-inf'), float('inf'),
                          turn, turn, make_move, fn)
        board, turn = make_move(board, turn, move, silent=True)
        turns += 1
        if turns > 500: break
    board[7] += sum(board[1:7])
    board[14] += sum(board[8:14])
    return board[7], board[14]


# ── Workers ──────────────────────────────────────────────────────────

def worker_match(args):
    a_label, a_kind, a_params, b_label, b_kind, b_params, depth = args
    aw, bw, ast, bst = 0, 0, 0, 0

    s1, s2 = simulate(a_kind, a_params, b_kind, b_params, depth)
    ast += s1; bst += s2
    if s1 > s2: aw += 1
    elif s2 > s1: bw += 1

    s1, s2 = simulate(b_kind, b_params, a_kind, a_params, depth)
    ast += s2; bst += s1
    if s2 > s1: aw += 1
    elif s1 > s2: bw += 1

    return {"a": a_label, "b": b_label, "aw": aw, "bw": bw,
            "ast": ast, "bst": bst, "sd": ast - bst}


def run_parallel(tasks, title):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"  {len(tasks)} matchups")
    print(f"{'='*75}")

    results = []
    t0 = time.perf_counter()
    done = 0

    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(worker_match, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            results.append(future.result())
            if done % 10 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - t0
                eta = elapsed / done * (len(tasks) - done)
                print(f"  [{done}/{len(tasks)}] ETA {eta:.0f}s", flush=True)

    print(f"  Done in {time.perf_counter() - t0:.1f}s")
    return results


def leaderboard(names, results):
    scores = {n: {"w": 0, "l": 0, "d": 0, "sf": 0, "sa": 0} for n in names}
    for r in results:
        a, b = r["a"], r["b"]
        dr = 2 - r["aw"] - r["bw"]
        scores[a]["w"] += r["aw"]; scores[a]["l"] += r["bw"]; scores[a]["d"] += dr
        scores[a]["sf"] += r["ast"]; scores[a]["sa"] += r["bst"]
        scores[b]["w"] += r["bw"]; scores[b]["l"] += r["aw"]; scores[b]["d"] += dr
        scores[b]["sf"] += r["bst"]; scores[b]["sa"] += r["ast"]

    board = []
    for n in names:
        s = scores[n]
        g = s["w"] + s["l"] + s["d"]
        board.append({"name": n, "w": s["w"], "l": s["l"], "d": s["d"],
                       "g": g, "wp": s["w"]/g*100 if g else 0,
                       "sd": s["sf"]-s["sa"], "avg": s["sf"]/g if g else 0})
    board.sort(key=lambda x: (x["w"], x["sd"]), reverse=True)

    print(f"\n  {'#':<3} {'Config':<42} {'W':>3} {'L':>3} {'D':>3} "
          f"{'Win%':>6} {'Diff':>6} {'AvgScr':>6}")
    print(f"  {'-'*75}")
    for rank, e in enumerate(board, 1):
        diff = f"{'+' if e['sd']>=0 else ''}{e['sd']}"
        print(f"  {rank:<3} {e['name']:<42} {e['w']:>3} {e['l']:>3} "
              f"{e['d']:>3} {e['wp']:>5.1f}% {diff:>6} {e['avg']:>6.1f}")
    return board


def main():
    depth = 10

    # ── From Phase 1 results: top 5 linear ───────────────────────────
    top_linear = {
        "Lin nw=5 pw=8":   ('lin', (5, 8.0)),
        "Lin nw=10 pw=5":  ('lin', (10, 5.0)),
        "Lin nw=10 pw=8":  ('lin', (10, 8.0)),
        "Lin nw=15 pw=8":  ('lin', (15, 8.0)),
        "Lin nw=5 pw=5":   ('lin', (5, 5.0)),
    }

    # ── Focused sigmoid grid (smaller: 4*4*4*4 = 256) ───────────────
    # Concentrate on regions that showed promise
    sigmoid_cands = {}
    for steep in [0.5, 0.8, 1.0, 1.5]:
        for nw in [50, 100, 200, 500]:
            for pw in [0.5, 1.0, 3.0, 5.0]:
                for ew in [0, 3, 5, 10]:
                    label = f"Sig s={steep} n={nw} p={pw} e={ew}"
                    sigmoid_cands[label] = ('sig', (steep, nw, pw, ew))

    # ── PHASE 2: Sigmoid vs best linear ──────────────────────────────
    best_lin = ('lin', (5, 8.0))
    tasks = []
    for label, (kind, params) in sigmoid_cands.items():
        tasks.append((label, kind, params, "Lin nw=5 pw=8", best_lin[0], best_lin[1], depth))

    results = run_parallel(tasks, f"PHASE 2: SIGMOID TUNING (256 configs vs Lin nw=5 pw=8)")

    # Rank sigmoids
    sig_ranked = []
    for r in results:
        sig_ranked.append({"label": r["a"], "w": r["aw"], "l": r["bw"],
                            "sd": r["sd"], "stones": r["ast"]})
    sig_ranked.sort(key=lambda x: (x["w"], x["sd"]), reverse=True)

    print(f"\n  Top 20 sigmoid configs:")
    print(f"  {'#':<3} {'Config':<42} {'W':>2} {'L':>2} {'Diff':>6}")
    print(f"  {'-'*58}")
    for i, r in enumerate(sig_ranked[:20], 1):
        diff = f"{'+' if r['sd']>=0 else ''}{r['sd']}"
        print(f"  {i:<3} {r['label']:<42} {r['w']:>2} {r['l']:>2} {diff:>6}")

    # ── PHASE 3: Finals round-robin ──────────────────────────────────
    finalists = dict(top_linear)  # top 5 linear
    for r in sig_ranked[:5]:      # top 5 sigmoid
        finalists[r["label"]] = sigmoid_cands[r["label"]]

    names = list(finalists.keys())
    rr_tasks = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            at, ap = finalists[a]
            bt, bp = finalists[b]
            rr_tasks.append((a, at, ap, b, bt, bp, depth))

    rr_results = run_parallel(rr_tasks, "PHASE 3: FINALS (Top 5 Linear vs Top 5 Sigmoid)")
    leaderboard(names, rr_results)


if __name__ == "__main__":
    main()

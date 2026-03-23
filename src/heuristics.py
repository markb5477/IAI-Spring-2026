import math


def evaluate_sigmoid(board, ai_player):
    """Sigmoid-based heuristic (current). Models win probability as a
    sigmoid centered at 24 seeds, with proximity-weighted pit scoring
    and an extra-turn bonus."""
    if ai_player == 2:
        my_nest, opp_nest = board[14], board[7]
        my_pits = sum(i * board[7 + i] for i in range(1, 7))
        opp_pits = sum(i * board[i] for i in range(1, 7))
    else:
        my_nest, opp_nest = board[7], board[14]
        my_pits = sum(i * board[i] for i in range(1, 7))
        opp_pits = sum(i * board[7 + i] for i in range(1, 7))

    def win_prob(n):
        return 1 / (1 + math.exp(-0.35 * (n - 24)))

    nest_score = (win_prob(my_nest) - win_prob(opp_nest)) * 100
    pit_score = (my_pits - opp_pits) * 0.8

    if ai_player == 1:
        extra_turn_bonus = sum(1 for i in range(1, 7) if board[i] == (7 - i))
    else:
        extra_turn_bonus = sum(1 for i in range(1, 7) if board[7 + i] == (7 - i))

    return nest_score + pit_score + extra_turn_bonus * 3


def evaluate_linear(board, ai_player):
    """Old linear heuristic. Simple weighted difference of nest and pit counts."""
    if ai_player == 2:
        nest_diff = board[14] - board[7]
        pit_diff = sum(board[8:14]) - sum(board[1:7])
    else:
        nest_diff = board[7] - board[14]
        pit_diff = sum(board[1:7]) - sum(board[8:14])

    return nest_diff * 20 + pit_diff

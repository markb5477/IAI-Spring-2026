import copy
import math


def evaluate_board(board, ai_player):
    """
    Scores the board from the AI's perspective using a linear heuristic.

    Uses weighted nest difference and pit scores to evaluate the board.
    Tuned weights (w_n=10, w_p=5) were determined through systematic
    parameter search and round-robin tournaments at depth 10, achieving
    a 61.1% win rate — the best overall configuration tested.

    Input:
        board (list[int]): The current Mancala board state represented as a
            list of 15 integers (index 0 unused, 1-6 are Player 1's pits,
            7 is Player 1's nest/store, 8-13 are Player 2's pits, 14 is
            Player 2's nest/store).
        ai_player (int): Which player the AI controls (1 or 2).

    Output:
        float: A heuristic score for the board. Positive values favor the AI,
            negative values favor the opponent.
    """
    if ai_player == 2:
        my_nest, opp_nest = board[14], board[7]
        my_pits = sum(i * board[7 + i] for i in range(1, 7))
        opp_pits = sum(i * board[i] for i in range(1, 7))
    else:
        my_nest, opp_nest = board[7], board[14]
        my_pits = sum(i * board[i] for i in range(1, 7))
        opp_pits = sum(i * board[7 + i] for i in range(1, 7))

    nest_score = (my_nest - opp_nest) * 10
    pit_score = (my_pits - opp_pits) * 5

    return nest_score + pit_score


def evaluate_board_sigmoid(board, ai_player):
    """
    Scores the board from the AI's perspective using a sigmoid-based heuristic.

    Uses a sigmoid function centered at 24 (half of 48 total stones) to model
    win probability. The steepest gradient is at the decision boundary where
    each additional stone matters most. Tuned parameters (k=0.8, w_nest=100,
    w_pit=0.5, w_extra=10) achieved a 55.6% win rate in round-robin finals.

    This heuristic requires deeper search (depth >= 10) to be competitive;
    at shallow depths the linear heuristic dominates.

    Input:
        board (list[int]): The current Mancala board state represented as a
            list of 15 integers (index 0 unused, 1-6 are Player 1's pits,
            7 is Player 1's nest/store, 8-13 are Player 2's pits, 14 is
            Player 2's nest/store).
        ai_player (int): Which player the AI controls (1 or 2).

    Output:
        float: A heuristic score for the board. Positive values favor the AI,
            negative values favor the opponent.
    """
    if ai_player == 2:
        my_nest, opp_nest = board[14], board[7]
        my_pits = sum(i * board[7 + i] for i in range(1, 7))
        opp_pits = sum(i * board[i] for i in range(1, 7))
    else:
        my_nest, opp_nest = board[7], board[14]
        my_pits = sum(i * board[i] for i in range(1, 7))
        opp_pits = sum(i * board[7 + i] for i in range(1, 7))

    def win_prob(n):
        return 1 / (1 + math.exp(-0.8 * (n - 24)))

    nest_score = (win_prob(my_nest) - win_prob(opp_nest)) * 100
    pit_score = (my_pits - opp_pits) * 0.5

    # Reward board states where extra-turn moves are available
    if ai_player == 1:
        extra_turn_bonus = sum(1 for i in range(1, 7) if board[i] == (7 - i))
    else:
        extra_turn_bonus = sum(1 for i in range(1, 7) if board[7 + i] == (7 - i))

    return nest_score + pit_score + extra_turn_bonus * 10


def minimax(board, depth, alpha, beta, current_player, ai_player, make_move):
    """
    Minimax search with alpha-beta pruning to find the optimal move.

    Input:
        board (list[int]): The current Mancala board state (see evaluate_board).
        depth (int): Maximum search depth remaining. Decremented each
            opponent turn; stays the same on extra turns (bonus moves).
        alpha (float): Best score the maximizing player can guarantee so far.
        beta (float): Best score the minimizing player can guarantee so far.
        current_player (int): The player whose turn it is (1 or 2).
        ai_player (int): Which player the AI controls (1 or 2).
        make_move (callable): Function with signature
            make_move(board, player, pit, silent=True) -> (new_board, next_player).
            Returns the resulting board and who moves next.

    Output:
        tuple[int, int | None]: A tuple of (best_score, best_move) where
            best_score is the heuristic evaluation and best_move is the
            pit index to play (or None if at a terminal/leaf node).
    """
    # Base case: depth exhausted or one side has no stones left (game over)
    if depth == 0 or sum(board[1:7]) == 0 or sum(board[8:14]) == 0:
        return evaluate_board(board, ai_player), None

    # Generate all legal moves for the current player (non-empty pits)
    moves = [i for i in (range(1, 7) if current_player == 1 else range(8, 14)) if board[i] > 0]

    # Determine the current player's nest index for move ordering
    nest = 7 if current_player == 1 else 14

    # Prioritize moves that land exactly in the player's nest (earning an extra turn),
    # since these branches tend to produce stronger lines and prune more aggressively.
    moves.sort(key=lambda m: (board[m] == (nest - m)), reverse=True)

    if current_player == ai_player:
        # Maximizing: find the move that produces the highest board evaluation.
        max_eval = float('-inf')
        best_move = moves[0]
        for move in moves:
            # Simulate the move on a copy of the board
            temp_board, next_p = make_move(board, current_player, move, silent=True)

            # Don't decrement depth on extra turns so the AI fully explores bonus moves.
            next_depth = depth if next_p == current_player else depth - 1

            # Recurse into the resulting game state
            eval_val, _ = minimax(temp_board, next_depth, alpha, beta, next_p, ai_player, make_move)

            # Update best move if this path scores higher
            if eval_val > max_eval:
                max_eval, best_move = eval_val, move

            # Alpha-beta pruning: cut off branches the minimizer would never allow
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        # Minimizing: assume the opponent picks the move that hurts the AI the most.
        min_eval = float('inf')
        best_move = moves[0]
        for move in moves:
            # Simulate the move on a copy of the board
            temp_board, next_p = make_move(board, current_player, move, silent=True)

            # Don't decrement depth on extra turns so the AI fully explores bonus moves.
            next_depth = depth if next_p == current_player else depth - 1

            # Recurse into the resulting game state
            eval_val, _ = minimax(temp_board, next_depth, alpha, beta, next_p, ai_player, make_move)

            # Update best move if this path scores lower (worse for the AI)
            if eval_val < min_eval:
                min_eval, best_move = eval_val, move

            # Alpha-beta pruning: cut off branches the maximizer would never allow
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        return min_eval, best_move

import copy


def evaluate_board(board, ai_player):
    """
    PRIORITY 1: Maximizing Nest difference (Captures/Steals are the fastest way here).
    PRIORITY 3: Defense is handled by Minimax naturally.
    """
    if ai_player == 2:
        return (board[14] - board[7]) * 20 + (sum(board[8:14]) - sum(board[1:7]))
    else:
        return (board[7] - board[14]) * 20 + (sum(board[1:7]) - sum(board[8:14]))


def minimax(board, depth, alpha, beta, current_player, ai_player, make_move):
    if depth == 0 or sum(board[1:7]) == 0 or sum(board[8:14]) == 0:
        return evaluate_board(board, ai_player), None

    moves = [i for i in (range(1, 7) if current_player == 1 else range(8, 14)) if board[i] > 0]
    nest = 7 if current_player == 1 else 14

    # PRIORITY 2: Move Ordering
    # We sort moves to check 'Extra Turn' scenarios first.
    # This speeds up Alpha-Beta Pruning significantly.
    moves.sort(key=lambda m: (board[m] == (nest - m)), reverse=True)

    if current_player == ai_player:
        max_eval = float('-inf')
        best_move = moves[0]
        for move in moves:
            temp_board, next_p = make_move(board, current_player, move, silent=True)
            next_depth = depth if next_p == current_player else depth - 1
            eval_val, _ = minimax(temp_board, next_depth, alpha, beta, next_p, ai_player, make_move)
            if eval_val > max_eval:
                max_eval, best_move = eval_val, move
            alpha = max(alpha, eval_val)
            if beta <= alpha: break
        return max_eval, best_move
    else:
        # This 'min' block is Priority 3 (Defense).
        # The AI assumes you will pick the move that hurts it the most.
        min_eval = float('inf')
        best_move = moves[0]
        for move in moves:
            temp_board, next_p = make_move(board, current_player, move, silent=True)
            next_depth = depth if next_p == current_player else depth - 1
            eval_val, _ = minimax(temp_board, next_depth, alpha, beta, next_p, ai_player, make_move)
            if eval_val < min_eval:
                min_eval, best_move = eval_val, move
            beta = min(beta, eval_val)
            if beta <= alpha: break
        return min_eval, best_move

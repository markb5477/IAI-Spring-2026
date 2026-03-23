import random
from src.ai_logic import minimax
from src.game import make_move


def make_random_agent():
    """Agent that picks a uniformly random legal move."""
    def agent(board, player):
        pits = range(1, 7) if player == 1 else range(8, 14)
        legal = [i for i in pits if board[i] > 0]
        return random.choice(legal)
    agent.name = "Random"
    return agent


def make_greedy_agent():
    """Agent that picks the move maximizing its own store after one ply."""
    def agent(board, player):
        pits = range(1, 7) if player == 1 else range(8, 14)
        nest = 7 if player == 1 else 14
        legal = [i for i in pits if board[i] > 0]
        return max(legal, key=lambda p: make_move(board, player, p, silent=True)[0][nest])
    agent.name = "Greedy"
    return agent


def make_minimax_agent(depth, evaluate_fn, label=None):
    """Agent that uses minimax with alpha-beta pruning at the given depth and heuristic."""
    def agent(board, player):
        _, move = minimax(board, depth, float('-inf'), float('inf'),
                          player, player, make_move, evaluate_fn)
        return move
    agent.name = label or f"Minimax-d{depth}"
    return agent

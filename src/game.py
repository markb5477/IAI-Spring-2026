import copy
import time
from src.ai_logic import minimax

# --- COLORS ---
C_BOLD, C_GREEN, C_YELLOW, C_CYAN, C_MAGENTA, C_WHITE, C_RESET = "\033[1m", "\033[1;32m", "\033[1;33m", "\033[1;36m", "\033[1;35m", "\033[1;37m", "\033[0m"
turn_count = 1

def get_initial_board():
    return [0, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]

def display_board(board):
    global turn_count
    print(f"\n{C_MAGENTA}{'='*25} TURN {turn_count} {'='*25}{C_RESET}")
    b = [f"{C_WHITE}{x}{C_YELLOW}" for x in board]
    print(f"{C_CYAN}{C_BOLD}                ◄  Direction      P 2's BOWLS            ")
    print(f"========================================================{C_YELLOW}")
    print(f"         _________13___12___11___10____9____8__________        ")
    print(f"        |          ↓    ↓    ↓    ↓    ↓    ↓          |       ")
    print(f"        |          ____________________________          |       ")
    print(f"        |         | {b[13]} ][ {b[12]} ][ {b[11]} ][ {b[10]} ][ {b[9]} ][ {b[8]} |         |")
    print(f"   P2   ► [{b[14]} ]------------------------------[{b[7]} ] ◄  P1   ")
    print(f"        |         | {b[1]} ][ {b[2]} ][ {b[3]} ][ {b[4]} ][ {b[5]} ][ {b[6]} |         |")
    print(f"         ‾‾‾‾‾‾‾‾‾ 1 ‾‾ 2 ‾‾ 3 ‾‾ 4 ‾‾ 5 ‾‾ 6 ‾‾‾‾‾‾‾‾‾        ")
    print(f"{C_CYAN}                          P 1's BOWLS      Direction  ►  {C_RESET}")
    turn_count += 1

def make_move(board, player, pit_index, silent=False):
    new_board = copy.deepcopy(board)
    pebbles = new_board[pit_index]
    new_board[pit_index] = 0
    current_pos = pit_index
    while pebbles > 0:
        current_pos = (current_pos % 14) + 1
        if player == 1 and current_pos == 14: continue
        if player == 2 and current_pos == 7: continue
        new_board[current_pos] += 1
        pebbles -= 1

    # STEAL LOGIC
    p1_range, p2_range = range(1, 7), range(8, 14)
    steal_happened = False
    if player == 1 and current_pos in p1_range and new_board[current_pos] == 1:
        opposite = 14 - current_pos
        if new_board[opposite] > 0:
            new_board[7] += new_board[opposite] + 1
            new_board[opposite], new_board[current_pos], steal_happened = 0, 0, True
            if not silent: print(f"{C_MAGENTA}!!! PLAYER STEAL !!!{C_RESET}")

    elif player == 2 and current_pos in p2_range and new_board[current_pos] == 1:
        opposite = 14 - current_pos
        if new_board[opposite] > 0:
            new_board[14] += new_board[opposite] + 1
            new_board[opposite], new_board[current_pos], steal_happened = 0, 0, True
            if not silent: print(f"{C_CYAN}!!! AI STEAL !!!{C_RESET}")

    # EXTRA TURN LOGIC
    extra_turn = (player == 1 and current_pos == 7) or (player == 2 and current_pos == 14)
    if extra_turn and not silent: print(f"{C_GREEN}EXTRA TURN!{C_RESET}")

    return new_board, (player if extra_turn else 3 - player)

def play_game():
    global turn_count
    turn_count, board = 1, get_initial_board()
    p1_name = input("Enter your name: ")
    turn = 1
    while not (sum(board[1:7]) == 0 or sum(board[8:14]) == 0):
        display_board(board)
        if turn == 1:
            try:
                choice = int(input(f"{p1_name}, choose bowl (1-6): "))
                if choice not in range(1, 7) or board[choice] == 0: continue
                board, turn = make_move(board, 1, choice)
            except ValueError: continue
        else:
            print("AI is calculating...")
            _, ai_choice = minimax(board, 10, float('-inf'), float('inf'), 2, 2, make_move)
            board, turn = make_move(board, 2, ai_choice)
            time.sleep(1)

    # Scoring
    board[7] += sum(board[1:7])
    board[14] += sum(board[8:14])
    display_board(board)
    print(f"FINAL SCORE - {p1_name}: {board[7]} | AI: {board[14]}")

if __name__ == "__main__":
    play_game()

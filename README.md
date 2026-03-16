# AI Techniques Used

## Minimax Algorithm
A decision-making algorithm used for turn-based games. It explores possible future game states by simulating moves for both players, assuming the opponent plays optimally. The AI maximizes its own score while minimizing the opponent's.

## Alpha-Beta Pruning
An optimization on top of Minimax that skips (prunes) branches of the game tree that cannot influence the final decision. This allows the AI to search deeper within the same time constraints.

## Heuristic Board Evaluation
A scoring function that estimates how favorable a board state is for the AI. It weights the nest (store) difference heavily and also considers the distribution of seeds across pits.

## Move Ordering
Moves are sorted to prioritize those likely to grant an extra turn (landing in the player's nest). Evaluating promising moves first improves the effectiveness of Alpha-Beta Pruning.


# Kalaha CLI

Source: <!-- TODO: add GitHub repo -->




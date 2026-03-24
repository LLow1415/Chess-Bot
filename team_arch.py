"""
team_alpha.py  —  Chess Bot using Minimax + Alpha-Beta Pruning
Heuristic: Material value

Install dependency:  pip install python-chess
"""

import chess

# ── Piece values (centipawns) ─────────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN:   2,
    chess.KNIGHT: 6,
    chess.BISHOP: 5,
    chess.ROOK:   7,
    chess.QUEEN:  8,
    chess.KING:   1,
}

# ── Heuristic ─────────────────────────────────────────────────────────────────
def evaluate(board: chess.Board) -> float:
    """

    Material:  Sum of piece values for White minus Black.

    Score > 0  =>  White is better.
    Score < 0  =>  Black is better.
    """
    if board.is_checkmate():
        # The side to move is in checkmate — they lose
        return -99999 if board.turn == chess.WHITE else 99999
    if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3) or board.is_fifty_moves():
        return 0
    if board.is_game_over():
        outcome = board.outcome()
        if outcome.winner is None:
            return 0  # draw (includes 5-fold repetition, 75-move rule, etc.)

    # ── Example Heuristic ─────────────────────────────────────────────────────────
    # The evaluation function counts the number of pieces White has minus the number
    # of pieces Black has, multiplied by a value of 1.

    # In the minimax algorithm, this evaluation is applied at the leaf nodes and
    # reflects the advantage of one player over the other. A positive score means
    # White is in a better position, while a negative score indicates that Black
    # is ahead.

    # The algorithm itself does not need to care about the player's colour,
    # because the evaluation function already represents the position from
    # both players' perspectives.

    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value # 10 white pieces, score = 10
        score -= len(board.pieces(piece_type, chess.BLACK)) * value # 5 black pieces, score = 5

    # Your code goes here

    # Reward Pawn Movement
    white_pawn = board.pieces(chess.PAWN, chess.WHITE)
    black_pawn = board.pieces(chess.PAWN, chess.BLACK)
    for pawn in white_pawn:
        if chess.square_rank(pawn) == 2:
            score += 1
        elif chess.square_rank(pawn)>2:
                score += 2
                if chess.square_rank(pawn)>3:
                    score += 1
                    if chess.square_rank(pawn)>4:
                        score += 1
                        if chess.square_rank(pawn)>5:
                            score += 1
                            if chess.square_rank(pawn)>6:
                                score += 4
    for pawn in black_pawn:
        if chess.square_rank(pawn) == 5:
            score -= 1
        elif chess.square_rank(pawn)<5:
                score -= 2
                if chess.square_rank(pawn)<4:
                    score -= 1
                    if chess.square_rank(pawn)<3:
                        score -= 1
                        if chess.square_rank(pawn)<2:
                            score -= 1
                            if chess.square_rank(pawn)<1:
                                score -= 4

    # Reward Check
    if board.is_check():
         return -8 if board.turn == chess.WHITE else 8

    last_move = board.peek()
    move_one = last_move.from_square
    move_two = last_move.to_square

    if chess.square_distance(move_one,move_two) > 1:
        return 3 if board.turn == chess.WHITE else -3
    if chess.square_manhattan_distance(move_one,move_two) > 1:
        return 3 if board.turn == chess.WHITE else -3
    if chess.square_knight_distance(move_one,move_two) > 2:
        return 3 if board.turn == chess.WHITE else -3

    return score

# ── Minimax with Alpha-Beta Pruning ───────────────────────────────────────────
# Simulates other team's worst-case response to a move, then get's your response to this worst-case move, continues until depth = 0, and returns best score by going back down the tree, eventually returning best move to function that called it
def minimax(board: chess.Board, depth: int,
            alpha: float, beta: float,
            maximizing: bool) -> float:
    """
    Standard Minimax search with Alpha-Beta cutoffs.
    maximizing=True means we are searching for the best move for White.
    """

    # Alpha = -inf
    # Beta = inf

    # When depth equals 0, the board is evaluated, and the score is returned, which sets the value of best = evaluate(board) when depth equals 1
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    # If Black, Start Here to see how white respons to move
    if maximizing:
        best = float('-inf') # Best Score = -inf
        for move in board.legal_moves: # Check every legal move possible
            board.push(move) # Push move to the board
            best = max(best, minimax(board, depth - 1, alpha, beta, False)) # Returns the highest value from best score, and minimax
            board.pop() # Remove last move from board
            alpha = max(alpha, best) # Alpha equals greater value between alpha and best
            if beta <= alpha:
                break       # Beta cutoff — opponent won't allow this path
        return best
    # If White, Start Here to see how white respons to move
    else:
        best = float('inf') # Best Score = inf
        for move in board.legal_moves:
            board.push(move)
            best = min(best, minimax(board, depth - 1, alpha, beta, True))
            board.pop()
            beta = min(beta, best)
            if beta <= alpha:
                break       # Alpha cutoff
        return best


# ── Entry point called by the tournament harness ──────────────────────────────
def get_next_move(board: chess.Board,
                  color: chess.Color,
                  depth: int = 3) -> chess.Move:
    """
    Return the best move for `color` from the current `board` position.
    DO NOT rename or change this signature — the harness calls it directly.
    """
    best_move  = None # End Best Move
    maximizing = (color == chess.WHITE) # If color = White, then max = true, else equals false so miniising
    best_score = float('-inf') if maximizing else float('inf') # if max = true, best_score = -inf, if minimising, then it equals inf

    b = board.copy()   # never modify the board passed in
    for move in b.legal_moves: # check every possible legal move that WHITE ot BLACK can currently make
        b.push(move) # pushes move onto the board copy temporarily
        score = minimax(b, depth - 1, float('-inf'), float('inf'), not maximizing) # Enter minimax with opposite of whatever maximising value is, Score = result of minimax with the board with the pushed move.
        b.pop() # Undoes last move pushed

        if maximizing and score > best_score: # If white, and current score is larger than best_score
            best_score, best_move = score, move
        elif not maximizing and score < best_score: # If black, and current score is lower than best_score
            best_score, best_move = score, move

    return best_move # Exit for loop, and return the best move based on score


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    b = chess.Board()
    move = get_next_move(b, chess.WHITE, depth=3)
    print(f"[team_arch] Opening move: {b.san(move)}")
    b.push(move)
    moveOne = b.peek()
    print(moveOne)

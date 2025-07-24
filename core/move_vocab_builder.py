#move_vocab_builder.py

import chess
import os
import torch
import pickle

VOCAB_PATH = "data/move_vocab/vocab.pkl"
def build_move_vocab():
    """
    Builds a comprehensive vocabulary of pseudo-legal chess moves as UCI strings,
    covering all piece moves from all squares on an empty board, plus explicit handling 
    of castling and promotions (including promotion captures). 
    
    Key points and caveats:
    - Starts by placing each piece type for both colors on every square of an empty board,
      generating all pseudo-legal moves from that square.
    - Pseudo-legal means moves are generated without considering checks or full legality.
    - Castling moves are manually added because they aren’t generated in the pseudo-legal moves 
      on an empty board.
    - Promotions are added in two steps:
       1) Normal forward promotions (e.g., e7e8q)
       2) Promotion captures (e.g., d7c8q) manually enumerated across adjacent files. This is because the previouse mechanism will not find the captures since there is only one piece on the board!
    - The function attempts to exhaustively cover all move possibilities relevant for move 
      prediction models or training, but due to complexity, some special cases might still be missing.
    - The resulting vocabulary maps UCI move strings to unique indices and vice versa.

    Warning: this is a Frankenstein's monster of heuristics, manual enumeration, and pseudo-legal 
    move generation, cobbled together to cover all moves you might see in typical chess datasets. 
    Use with care and double-check if certain rare moves or edge cases appear missing.

    Returns:
        uci_to_index (dict): Mapping from UCI move strings to unique integer indices.
        index_to_uci (dict): Reverse mapping from indices to UCI move strings.
    """
        
    uci_to_index = {}
    index_to_uci = {}
    index = 0

    colors = [chess.BLACK, chess.WHITE]
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    for color in colors:
        for from_square in chess.SQUARES:
            for piece_type in piece_types:
                board = chess.Board.empty()                
                board.set_piece_at(from_square, chess.Piece(piece_type, color))
                board.turn = color

                for move in board.generate_pseudo_legal_moves():
                    if move.from_square != from_square:
                        continue
                    uci = move.uci()
                    if uci not in uci_to_index:
                        uci_to_index[uci] = index
                        index_to_uci[index] = uci
                        index += 1

    # Add castling moves explicitly
    castling_moves = ['e1g1', 'e1c1', 'e8g8', 'e8c8']
    for uci in castling_moves:
        if uci not in uci_to_index:
            uci_to_index[uci] = index
            index_to_uci[index] = uci
            index += 1

    # === Manually Add All Possible Promotion Moves ===
    promotion_pieces = ['q', 'r', 'b', 'n']
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
 
    # --- Promotion captures ---

    # Helper to get adjacent files for captures
    def adjacent_files(f):
        idx = files.index(f)
        adj = []
        if idx > 0:
            adj.append(files[idx - 1])
        if idx < 7:
            adj.append(files[idx + 1])
        return adj

    # White promotion captures: from rank 7 to rank 8 on adjacent files
    for file in files:
        from_rank = '7'
        to_rank = '8'
        from_square = file + from_rank
        for capture_file in adjacent_files(file):
            to_square = capture_file + to_rank
            for promo in promotion_pieces:
                uci = from_square + to_square + promo
                if uci not in uci_to_index:
                    uci_to_index[uci] = index
                    index_to_uci[index] = uci
                    index += 1

    # Black promotion captures: from rank 2 to rank 1 on adjacent files
    for file in files:
        from_rank = '2'
        to_rank = '1'
        from_square = file + from_rank
        for capture_file in adjacent_files(file):
            to_square = capture_file + to_rank
            for promo in promotion_pieces:
                uci = from_square + to_square + promo
                if uci not in uci_to_index:
                    uci_to_index[uci] = index
                    index_to_uci[index] = uci
                    index += 1

    print(f"Total moves in vocabulary: {index}")
    return uci_to_index, index_to_uci

def build_move_component_indices(index_to_uci):
    from_ids = []
    to_ids = []
    promo_ids = []

    for i in range(len(index_to_uci)):
        uci = index_to_uci[i]
        move = chess.Move.from_uci(uci)
        from_ids.append(move.from_square)
        to_ids.append(move.to_square)
        if move.promotion is None:
            promo_ids.append(0)
        else:
            mapping = {chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
            promo_ids.append(mapping[move.promotion])

    return torch.tensor(from_ids), torch.tensor(to_ids), torch.tensor(promo_ids)


def load_or_build_vocab():
    os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, "rb") as f:
            data = pickle.load(f)
            print("Loaded move vocab from disk.")
            return data["uci_to_index"], data["index_to_uci"], data["from_ids"], data["to_ids"], data["promo_ids"]
    
    uci_to_index, index_to_uci = build_move_vocab()
    from_ids, to_ids, promo_ids = build_move_component_indices(index_to_uci)

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump({
            "uci_to_index": uci_to_index,
            "index_to_uci": index_to_uci,
            "from_ids": from_ids,
            "to_ids": to_ids,
            "promo_ids": promo_ids
        }, f)
        print("Saved move vocab to disk.")

    return uci_to_index, index_to_uci, from_ids, to_ids, promo_ids

if __name__ == "__main__":
    uci_to_index, index_to_uci = build_move_vocab()
    print("Sample moves:")
    for i in range(20):
        print(f"{i}: {index_to_uci[i]}")

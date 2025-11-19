import numpy as np
import json
import socket
import time
from numba import njit

# --- サーバ定義 ---
BUFSIZE = 1024
serverName = "localhost"
serverPort = 4444

# --- 定数定義 ---
LION = 1
GIRAFFE = 2
ELEPHANT = 3
CHICK = 4
CHICKEN = 5

PLAYER1 = 1
PLAYER2 = 2

WIN_SCORE = 10000

# [dummy, LION, GIRAFFE, ELEPHANT, CHICK, CHICKEN]
KOMA_VALUE = np.array([0, 1000, 50, 45, 20, 40], dtype=np.int16)

PIECE_NAME_P1 = ["", "l1", "g1", "e1", "c1", "h1"]
PIECE_NAME_P2 = ["", "l2", "g2", "e2", "c2", "h2"]

# TTの中身: [hash, score, depth, best_from, best_to, best_is_drop, flag]

# TT用フラグ
FLAG_EXACT = 0
FLAG_LOWER = 1
FLAG_UPPER = 2

# TTテーブルサイズ
TT_SIZE_BITS = 20
TT_SIZE = 1 << TT_SIZE_BITS
TT_MASK = TT_SIZE - 1

# --- Zobrist Hashing 用の乱数テーブル生成 ---
np.random.seed(42)
# 盤面: [12マス][駒の種類]
Z_BOARD = np.random.randint(0, 2**62, (12, 12), dtype=np.int64)
# 持ち駒: [player 0-1][piece 1-5][count 0-2]
Z_HAND = np.random.randint(0, 2**62, (2, 6, 3), dtype=np.int64)
# 手番
Z_TURN = np.random.randint(0, 2**62, 1, dtype=np.int64)[0]


@njit
def compute_hash(board, hand, player):
    """
    Zobrist hashを計算する関数

    Parameters
    ----------
    board : np.ndarray
        盤面の配列 
    hand : np.ndarray
        持ち駒の配列
    player : int
        手番 (1 or 2)
    
    Returns
    ----------
    h : int
        計算されたハッシュ値
    """
    h = 0
    # 盤面
    for idx in range(12):
        p = board[idx]
        if p != 0:
            # pは -5 ~ 5 配列index用に+6 
            h ^= Z_BOARD[idx, p + 6]
    
    # 持ち駒
    for pl in range(2):
        for k in range(1, 6):
            cnt = hand[pl, k]
            if cnt > 0:
                h ^= Z_HAND[pl, k, cnt]
    
    # 手番  Player2ならXOR
    if player == 2:
        h ^= Z_TURN
        
    return h

@njit
def idx_to_xy(idx):
    return idx % 3, idx // 3

@njit
def xy_to_idx(x, y):
    return x + y * 3

@njit
def get_legal_moves_jit(board, hand, player):
    """
    合法手を取得する関数

    Parameters
    ----------
    board : np.ndarray
        盤面の配列
    hand : np.ndarray
        持ち駒の配列
    player : int
        手番 (1 or 2)
    
    Returns
    ----------
    moves : np.ndarray
        合法手の配列 (from, to, is_drop)
    """

    moves_f = []
    moves_t = []
    moves_d = []
    
    dir_val = 1 if player == 1 else -1
    
    # --- 盤上の駒 ---
    for idx in range(12):
        piece = board[idx]
        if piece == 0:
            continue
        
        # 自分の駒か確認
        if (player == 1 and piece > 0) or (player == 2 and piece < 0):
            p_type = abs(piece)
            cx, cy = idx_to_xy(idx)
            
            # 移動定義
            dxs = np.empty(0, dtype=np.int8)
            dys = np.empty(0, dtype=np.int8)

            if p_type == LION:
                dxs = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int8)
                dys = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int8)
            elif p_type == GIRAFFE:
                dxs = np.array([-1, 1, 0, 0], dtype=np.int8)
                dys = np.array([0, 0, -1, 1], dtype=np.int8)
            elif p_type == CHICK:
                dxs = np.array([0], dtype=np.int8)
                dys = np.array([-1 * dir_val], dtype=np.int8)
            elif p_type == CHICKEN:
                dxs = np.array([-1, 0, 1, -1, 1, 0], dtype=np.int8)
                dys = np.array([-1*dir_val, -1*dir_val, -1*dir_val, 0, 0, 1*dir_val], dtype=np.int8)
            elif p_type == ELEPHANT:
                dxs = np.array([-1, -1, 1, 1], dtype=np.int8)
                dys = np.array([-1, 1, -1, 1], dtype=np.int8)
            
            for i in range(len(dxs)):
                tx = cx + dxs[i]
                ty = cy + dys[i]
                
                if tx < 0 or tx > 2 or ty < 0 or ty > 3:
                    continue
                
                t_idx = xy_to_idx(tx, ty)
                # 行き先に自分の駒がないか
                if board[t_idx] * dir_val > 0:
                    continue
                
                moves_f.append(idx)
                moves_t.append(t_idx)
                moves_d.append(0)

    # --- 持ち駒 ---
    hand_row = 0 if player == 1 else 1
    for p_type in range(1, 6):
        if hand[hand_row, p_type] > 0:
            for t_idx in range(12):
                if board[t_idx] == 0:
                    moves_f.append(p_type)
                    moves_t.append(t_idx)
                    moves_d.append(1)

    count = len(moves_f)
    res = np.zeros((count, 3), dtype=np.int64)
    for i in range(count):
        res[i, 0] = moves_f[i]
        res[i, 1] = moves_t[i]
        res[i, 2] = moves_d[i]
        
    return res

@njit
def make_move_jit(board, hand, move, player, current_hash):
    """
    一手指す関数
    盤面のコピーを作らないので、undo_make_move_jitで元に戻して使ってね

    Parameters
    ----------
    board : np.ndarray
        盤面の配列
    hand : np.ndarray
        持ち駒の配列
    move : tuple
        指す手 (from, to, is_drop)
    player : int
        手番 (1 or 2)
    current_hash : int
        現在のハッシュ値

    Returns
    ----------
    captured : int
        捕獲した駒 (0ならなし)
    is_promoted : int
        成りフラグ (1なら成り)
    next_hash : int
        次のハッシュ値
    """
    f, t, is_drop = move
    owner = 0 if player == 1 else 1
    dir_val = 1 if player == 1 else -1
    
    captured = 0
    is_promoted = 0
    
    # ハッシュ更新用
    next_hash = current_hash
    next_hash ^= Z_TURN 

    if is_drop == 1:
        # 持ち駒を減らす
        hand[owner, f] -= 1
        next_hash ^= Z_HAND[owner, f, hand[owner, f] + 1] # 元の数を除去
        next_hash ^= Z_HAND[owner, f, hand[owner, f]]     # 新しい数を追加
        
        # 盤面に置く
        board[t] = f * dir_val
        next_hash ^= Z_BOARD[t, (f * dir_val) + 6]
        
    else:
        # 移動元の駒
        piece = board[f]
        target = board[t]
        
        # 移動元のハッシュ除去
        next_hash ^= Z_BOARD[f, piece + 6]
        board[f] = 0

        # 捕獲処理
        if target != 0:
            next_hash ^= Z_BOARD[t, target + 6]
            captured = target
            cap_type = abs(target)
            if cap_type == CHICKEN:
                cap_type = CHICK
            
            hand[owner, cap_type] += 1
            next_hash ^= Z_HAND[owner, cap_type, hand[owner, cap_type] - 1]
            next_hash ^= Z_HAND[owner, cap_type, hand[owner, cap_type]]

        # 成り判定
        p_abs = abs(piece)
        if p_abs == CHICK:
            _, ty = idx_to_xy(t)
            if (player == 1 and ty == 0) or (player == 2 and ty == 3):
                is_promoted = 1
                piece = CHICKEN * dir_val
        
        # 移動先へ配置
        board[t] = piece
        next_hash ^= Z_BOARD[t, piece + 6]

    return captured, is_promoted, next_hash

@njit
def undo_make_move_jit(board, hand, move, player, captured, is_promoted):
    """
    指した手を戻す関数

    Parameters
    ----------
    board : np.ndarray
        盤面の配列
    hand : np.ndarray
        持ち駒の配列
    move : tuple
        指した手 (from, to, is_drop)
    player : int
        手番 (1 or 2)
    captured : int
        捕獲した駒 (0ならなし)
    is_promoted : int
        成りフラグ (1なら成り)
    """
    f, t, is_drop = move
    owner = 0 if player == 1 else 1
    dir_val = 1 if player == 1 else -1

    if is_drop == 1:
        board[t] = 0
        hand[owner, f] += 1
    else:
        moved_piece = board[t]
        if is_promoted == 1:
            moved_piece = CHICK * dir_val
        
        board[f] = moved_piece
        board[t] = captured
        
        if captured != 0:
            cap_type = abs(captured)
            if cap_type == CHICKEN:
                cap_type = CHICK
            hand[owner, cap_type] -= 1

@njit
def get_repetition_count(history, current_hash):
    """
    履歴の中に current_hash が何回登場したか数える
    """
    count = 0
    for h in history:
        if h == current_hash:
            count += 1
    return count

@njit
def win_check_jit(board, hand, player):
    """
    勝敗判定を行う関数

    Parameters
    ----------
    board : np.ndarray
        盤面の配列
    hand : np.ndarray
        持ち駒の配列
    player : int
        手番 (1 or 2)

    Returns
    ----------
    result : int
        勝敗結果 (WIN_SCORE: 勝ち, -WIN_SCORE: 負け, 0: 継続)
    """
    op = 2 if player == 1 else 1
    dir_val = 1 if player == 1 else -1
    my_goal = 0 if player == 1 else 3
    op_goal = 3 if player == 1 else 0

    my_lion_pos = -1
    op_lion_pos = -1
    
    for i in range(12):
        if board[i] == LION * dir_val:
            my_lion_pos = i
        elif board[i] == LION * -dir_val:
            op_lion_pos = i
            
    if my_lion_pos == -1: return -WIN_SCORE
    if op_lion_pos == -1: return WIN_SCORE

    # トライ判定
    _, my_y = idx_to_xy(my_lion_pos)
    if my_y == my_goal:
        op_moves = get_legal_moves_jit(board, hand, op)
        is_captured = False
        for i in range(len(op_moves)):
            if op_moves[i, 1] == my_lion_pos:
                is_captured = True
                break
        
        if not is_captured:
            return WIN_SCORE
        else:
            return -WIN_SCORE

    _, op_y = idx_to_xy(op_lion_pos)
    if op_y == op_goal:
        my_moves = get_legal_moves_jit(board, hand, player)
        is_captured = False
        for i in range(len(my_moves)):
            if my_moves[i, 1] == op_lion_pos:
                is_captured = True
                break
        if not is_captured:
            return -WIN_SCORE
        else:
            return WIN_SCORE

    return 0

@njit
def eval_board_jit(board, hand, player):
    """
    評価関数

    Parameters
    ----------
    board : np.ndarray
        盤面の配列
    hand : np.ndarray
        持ち駒の配列
    player : int
        手番 (1 or 2)

    Returns
    ----------
    eval_value : int
        評価値
    """

    my_points = 0
    op_points = 0
    
    dir_val = 1 if player == 1 else -1
    hand_idx = 0 if player == 1 else 1
    
    for i in range(12):
        p = board[i]
        if p == 0: continue
        if p * dir_val > 0:
            my_points += KOMA_VALUE[abs(p)]
        else:
            op_points += KOMA_VALUE[abs(p)]
            
    for k in range(1, 6):
        n = hand[hand_idx, k]
        if n > 0:
            my_points += KOMA_VALUE[k] * n
        n_op = hand[1 - hand_idx, k]
        if n_op > 0:
            op_points += KOMA_VALUE[k] * n_op
            
    return my_points - op_points

@njit
def alpha_beta_jit(board, hand, depth, player, alpha, beta, current_hash, tt, history):
    """
    αβ法による探索をする関数
    negamaxだと思います。

    Parameters
    ----------
    board : np.ndarray
        盤面の配列
    hand : np.ndarray
        持ち駒の配列
    depth : int
        探索深さ
    player : int
        手番 (1 or 2)
    alpha : int
        α値
    beta : int
        β値
    current_hash : int
        現在のハッシュ値
    tt : np.ndarray
        トランスポジションテーブル

    Returns
    ----------
    score : int
        評価値
    best_move_f : int
        最善手のfrom
    best_move_t : int
        最善手のto
    best_move_d : int
        最善手のis_drop
    """

    if get_repetition_count(history, current_hash) >= 2:
        return 0, -1, -1, 0

    tt_index = current_hash & TT_MASK
    entry = tt[tt_index]
    
    if entry[0] == current_hash:
        if entry[2] >= depth:
            score = entry[1]
            flag = entry[6]
            
            # カットされなかった場合
            if flag == FLAG_EXACT:
                return score, entry[3], entry[4], entry[5]
            
            # βカットが発生した場合
            elif flag == FLAG_LOWER:
                if score >= beta:
                    return score, entry[3], entry[4], entry[5]
                if score > alpha:
                    alpha = score
            
            # 一番高いα
            elif flag == FLAG_UPPER:
                if score <= alpha:
                    return score, entry[3], entry[4], entry[5]
                if score < beta:
                    beta = score

    # 決着判定
    wc = win_check_jit(board, hand, player)
    if wc != 0:
        tt[tt_index] = np.array([current_hash, wc, 99, 0, 0, 0, FLAG_EXACT], dtype=np.int64)
        return wc, -1, -1, 0
        
    if depth == 0:
        val = eval_board_jit(board, hand, player)
        tt[tt_index] = np.array([current_hash, val, 0, 0, 0, 0, FLAG_EXACT], dtype=np.int64)
        return val, -1, -1, 0

    original_alpha = alpha
    best_score = -999999
    best_move_f, best_move_t, best_move_d = -1, -1, 0
    
    moves = get_legal_moves_jit(board, hand, player)
    
    if len(moves) == 0:
        return -WIN_SCORE, -1, -1, 0
        
    for i in range(len(moves)):
        move = moves[i]
        
        captured, is_promoted, next_hash = make_move_jit(board, hand, move, player, current_hash)
        
        next_player = 2 if player == 1 else 1
        score, _, _, _ = alpha_beta_jit(board, hand, depth - 1, next_player, -beta, -alpha, next_hash, tt, history)
        score = -score
        
        undo_make_move_jit(board, hand, move, player, captured, is_promoted)
        
        if score > best_score:
            best_score = score
            if score > alpha:
                alpha = score
                best_move_f = move[0]
                best_move_t = move[1]
                best_move_d = move[2]
                
        if alpha >= beta:
            tt[tt_index] = np.array([
                current_hash, alpha, depth, move[0], move[1], move[2], FLAG_LOWER
            ], dtype=np.int64)
            return alpha, move[0], move[1], move[2]

    flag = FLAG_EXACT
    if best_score <= original_alpha:
        flag = FLAG_UPPER
    
    tt[tt_index] = np.array([
        current_hash, alpha, depth, best_move_f, best_move_t, best_move_d, flag
    ], dtype=np.int64)

    return alpha, best_move_f, best_move_t, best_move_d

@njit
def get_diff_board(b_prev, h_prev, b_curr, h_curr, player):
    """
    盤面の差分を取得する関数
    Parameters
    ----------
    old_board : np.ndarray
        以前の盤面の配列
    old_hand : np.ndarray
        以前の持ち駒の配列
    new_board : np.ndarray
        新しい盤面の配列
    new_hand : np.ndarray
        新しい持ち駒の配列

    Returns
    ----------
    from : int
        指した手のfrom
    to : int
        指した手のto
    is_drop : int
        指した手のis_drop
    """
    dir_val = 1 if player == 1 else -1
    hand_row = 0 if player == 1 else 1

    # 持ち駒の変化を確認
    for k in range(1, 6):
        if h_prev[hand_row, k] > h_curr[hand_row, k]:
            # 持ち駒を打った
            for i in range(12):
                if b_prev[i] == 0 and b_curr[i] == k * dir_val:
                    return k, i, 1
    
    # 盤面の変化を確認
    from_idx = -1
    to_idx = -1

    for i in range(12):
        p_prev = b_prev[i]
        p_curr = b_curr[i]

        # 同じだったら割愛
        if p_prev == p_curr:
            continue

        # 移動元 あったのになくなった
        if p_prev * dir_val > 0 and p_curr == 0:
            from_idx = i
        
        # 移動先 なかったのに自分の駒がある
        if p_curr * dir_val > 0:
            to_idx = i
    
    return from_idx, to_idx, 0

@njit
def get_lion_capture_move(board, hand, player):
    """
    相手のライオンをとれるなら取る手を返す関数

    Parameters
    ----------
    board : np.ndarray
        盤面の配列
    hand : np.ndarray
        持ち駒の配列
    player : int
        手番 (1 or 2)

    Returns
    ----------
    has_capture : bool
        捕獲できる手があるか
    from : int
        指す手のfrom
    to : int
        指す手のto
    is_drop : int
        指す手のis_drop
    """
    moves = get_legal_moves_jit(board, hand, player)
    
    target_lion = -LION if player == 1 else LION
    
    for i in range(len(moves)):
        f = moves[i][0]
        t = moves[i][1]
        d = moves[i][2]
        
        if d == 1:
            continue
            
        # 移動先に相手のライオンがいるか
        if board[t] == target_lion:
            return True, f, t, d
            
    return False, -1, -1, 0

# ここからはjitが使えない
# というか使う必要がない

def parse_board(board_dict):
    b = np.zeros(12, dtype=np.int8)
    hand = np.zeros((2, 6), dtype=np.int8)
    piece_map = {'l': LION, 'g': GIRAFFE, 'e': ELEPHANT, 'c': CHICK, 'h': CHICKEN}
    
    def masu_to_idx(masu):
        return (ord(masu[0]) - ord('A')) + (int(masu[1]) - 1) * 3

    for pos, piece in board_dict.items():
        p = piece[0]
        owner = int(piece[1])
        val = piece_map[p]
        
        if pos.startswith(('D', 'E')):
            row = 0 if pos.startswith('D') else 1
            hand[row][val] += 1
        else:
            idx = masu_to_idx(pos)
            b[idx] = val if owner == 1 else -val
    return b, hand

def unparse_move(f, t, is_drop, board_json, player):
    if is_drop == 1:
        if player == 1:
            p = PIECE_NAME_P1[f]
            src = [k for k, v in board_json.items() if v == p and k.startswith("D")][0]
        else:
            p = PIECE_NAME_P2[f]
            print(p)
            print(board_json)
            src = [k for k, v in board_json.items() if v == p and k.startswith("E")][0]

        
        dest = chr(t % 3 + ord('A')) + str(t // 3 + 1)
        return src, dest
    else:
        src = chr(f % 3 + ord('A')) + str(f // 3 + 1)
        dest = chr(t % 3 + ord('A')) + str(t // 3 + 1)
        return src, dest

def board_print(board, hand):
    print("--- Hand ---")
    print(f"P1: {hand[0][1:]}")
    print(f"P2: {hand[1][1:]}")
    print("--- Board ---")
    for y in range(4):
        line = ""
        for x in range(3):
            idx = x + y*3
            p = board[idx]
            s = " . "
            if p != 0:
                owner = "1" if p > 0 else "2"
                ptype = ["?", "L", "G", "E", "C", "H"][abs(p)]
                s = f"{ptype}{owner} "
            line += s
        print(line)
    print("-------------")

def main():
    tt_table = np.zeros((TT_SIZE, 7), dtype=np.int64)

    hash_history = []

    board_history = []

    # ダミーを走らせてJITコンパイル
    print("JIT Compiling...")
    _b = np.zeros(12, dtype=np.int8)
    _h = np.zeros((2, 6), dtype=np.int8)
    _hash = compute_hash(_b, _h, 1)
    _hist = np.array([], dtype=np.int64)
    alpha_beta_jit(_b, _h, 1, 1, -10000, 10000, _hash, tt_table, _hist)
    get_lion_capture_move(_b, _h, 1)
    print("Done.")

    # サーバー接続
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((serverName, serverPort))
    except ConnectionRefusedError:
        print("エラー: サーバーに接続できません。")
        exit()

    # メッセージ送受信関数
    def sendMessage(cmd):
        s.send(f"{cmd}\n".encode())
        time.sleep(0.05)
        msg = ""
        while True:
            chunk = s.recv(BUFSIZE).decode()
            msg += chunk
            if msg.endswith("\n"):
                break
        return msg.strip()
    
    # ここからゲームスタート
    msg = s.recv(BUFSIZE).rstrip().decode()
    print(f"Server: {msg}")
    myTurn = 1 if "Player1" in msg else 2
    print(f"あなたは Player{myTurn} です。")

    board_old_json = sendMessage("initboardjson")
    board_j_o = json.loads(board_old_json)

    DEPTH = 10

    while True:
        current_turn_msg = sendMessage("turn")
        current_turn = 0
        if current_turn_msg == "Player1":
            current_turn = 1
        elif current_turn_msg == "Player2":
            current_turn = 2
        
        if myTurn == current_turn:
            print(f"\nPlayer{myTurn} のターンです。")
            
            boardjson_str = sendMessage("boardjson")
            board_j = json.loads(boardjson_str)
            board, hand = parse_board(board_j)

            board_history.append((board.copy(), hand.copy()))

            board_print(board, hand)
            
            current_hash = compute_hash(board, hand, myTurn)

            hash_history.append(current_hash)

            can_capture, f, t, d = get_lion_capture_move(board, hand, myTurn)
            if can_capture:
                print("相手のライオンをとれます。")
                src, dest = unparse_move(f, t, d, board_j, myTurn)
                print(f"最善手: {src} -> {dest}")
                
                resp = sendMessage(f"mv {src} {dest}")
                print(f"サーバからの応答: {resp}")
                
                boardjson_str = sendMessage("boardjson")
                board_j_o = json.loads(boardjson_str)
                continue

            if hash_history.count(current_hash) >= 3:
                print("三回同一局面に到達しました。引き分けです。")
            
            print(f"思考中... (深さ: {DEPTH})")
            start_time = time.time()

            history_np = np.array(hash_history[:-1], dtype=np.int64)
            
            score, f, t, d = alpha_beta_jit(board, hand, DEPTH, myTurn, -999999, 999999, current_hash, tt_table, history_np)
            
            elapsed = time.time() - start_time
            print(f"完了: {elapsed:.4f}秒, 評価値: {score}")

            if f == -1:
                moves = get_legal_moves_jit(board, hand, myTurn)
                if len(moves) > 0:
                    print("千日手っぽくなるので履歴を無視して探索します")
                    dummy_hist = np.array([], dtype=np.int64)
                    score, f, t, d = alpha_beta_jit(board, hand, DEPTH, myTurn, -999999, 999999, current_hash, tt_table, dummy_hist)

            if f == -1:
                print("指せる手がありません")
                break
            
            src, dest = unparse_move(f, t, d, board_j, myTurn)
            print(f"最善手: {src} -> {dest}")
            
            resp = sendMessage(f"mv {src} {dest}")
            print(f"サーバからの応答: {resp}")

            boardjson_str = sendMessage("boardjson")
            board_j_o = json.loads(boardjson_str)
        else:
            # 相手番
            time.sleep(0.5)
            boardjson_str = sendMessage("boardjson")
            board_j = json.loads(boardjson_str)

            if len(board_history) > 0:
                prev_b, prev_h = board_history[-1]

                op_player = 2 if myTurn == 1 else 1

                f, t, d = get_diff_board(prev_b, prev_h, *parse_board(board_j), op_player)

                if f != -1:
                    src, dest = unparse_move(f, t, d, board_j_o, op_player)
                    print(f"\nPlayer{op_player} の手番です。")
                    print(f"相手の指した手: {src} -> {dest}")

            board, hand = parse_board(board_j)
            wc = win_check_jit(board, hand, 2 if myTurn == 1 else 1)
            if wc == WIN_SCORE:
                print("\nあなたの負けです！")
                break
            elif wc == -WIN_SCORE:
                print("\nあなたの勝ちです。")
                break

    s.close()

if __name__ == "__main__":
    main()
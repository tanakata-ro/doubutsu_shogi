from flask import Flask, render_template, request, jsonify
import numpy as np
import game_logic as ai
from numba import njit

app = Flask(__name__)

# --- サーバー起動時の初期化 ---
print("Initializing AI & JIT compilation...")

# TTテーブルの確保
TT_TABLE = np.zeros((ai.TT_SIZE, 7), dtype=np.int64)

# ダミーデータでJITコンパイルを済ませる
_b = np.zeros(12, dtype=np.int8)
_h = np.zeros((2, 6), dtype=np.int8)
_hash = ai.compute_hash(_b, _h, 1)
_hist_buff = np.zeros(200, dtype=np.int64) # 履歴バッファ
_hist_len = 0                              # 履歴の長さ

# 1. 合法手生成のコンパイル
ai.get_legal_moves_jit(_b, _h, 1)

# 2. 探索関数のコンパイル (修正箇所: 引数に _hist_len を追加)
ai.alpha_beta_jit(_b, _h, 1, 1, -10000, 10000, _hash, TT_TABLE, _hist_buff, _hist_len)

print("Ready!")

# 初期配置定義
INITIAL_BOARD = [
    -2, -1, -3, # 相手 G L E (Player 2)
     0, -4,  0, # 相手 . C .
     0,  4,  0, # 自分 . C .
     3,  1,  2  # 自分 E L G (Player 1)
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_game():
    """ゲーム開始処理"""
    data = request.json
    human_player = int(data.get('human_player', 1))
    
    board = np.array(INITIAL_BOARD, dtype=np.int8)
    hand = np.zeros((2, 6), dtype=np.int8)
    
    if human_player == 1:
        return jsonify({
            'status': 'playing',
            'board': board.tolist(),
            'hand': hand.tolist(),
            'turn': 1
        })
    else:
        ai_player = 1
        board, hand, score, move_desc, game_status = run_ai_turn(board, hand, ai_player)
        return jsonify({
            'status': game_status,
            'board': board.tolist(),
            'hand': hand.tolist(),
            'turn': 2,
            'message': f"AI Move: {move_desc}"
        })

@app.route('/move', methods=['POST'])
def make_move():
    """人間の手を受信 -> AIが打ち返す"""
    data = request.json
    
    board = np.array(data['board'], dtype=np.int8)
    hand = np.array(data['hand'], dtype=np.int8)
    human_player = int(data['human_player'])
    ai_player = 2 if human_player == 1 else 1
    
    move_f = int(data['move']['from'])
    move_t = int(data['move']['to'])
    move_d = int(data['move']['drop'])

    # 1. 合法手チェック
    if not is_legal_move(board, hand, human_player, move_f, move_t, move_d):
        return jsonify({'status': 'error', 'message': 'その手は反則です (Illegal Move)'})

    # 2. 人間の手を反映
    current_hash = ai.compute_hash(board, hand, human_player)
    move_tuple = (move_f, move_t, move_d)
    captured, is_promoted, _ = ai.make_move_jit(board, hand, move_tuple, human_player, current_hash)

    # 3. 人間の勝利判定
    win_result = ai.win_check_jit(board, hand, human_player)
    if win_result == ai.WIN_SCORE:
        return jsonify({
            'status': 'win', 
            'board': board.tolist(), 
            'hand': hand.tolist(),
            'message': 'あなたの勝ちです！'
        })

    # 4. AIのターン
    board, hand, score, move_desc, ai_status = run_ai_turn(board, hand, ai_player)

    if ai_status == 'win':
        return jsonify({
            'status': 'lose',
            'board': board.tolist(),
            'hand': hand.tolist(),
            'message': 'あなたの負けです...'
        })
    
    return jsonify({
        'status': 'playing',
        'board': board.tolist(),
        'hand': hand.tolist(),
        'score': int(score),
        'message': f"AI: {move_desc}"
    })

def is_legal_move(board, hand, player, f, t, d):
    moves = ai.get_legal_moves_jit(board, hand, player)
    for i in range(len(moves)):
        if moves[i][0] == f and moves[i][1] == t and moves[i][2] == d:
            return True
    return False

def run_ai_turn(board, hand, player):
    """AI思考ロジック"""
    current_hash = ai.compute_hash(board, hand, player)
    
    # --- 修正箇所: 履歴バッファと長さを定義 ---
    # Webアプリの一手ごとなので、過去の履歴は今回は空(0)として扱います
    # (千日手を厳密にやるならJSから履歴配列を送る必要がありますが、まずはこれで動きます)
    history_buffer = np.zeros(200, dtype=np.int64)
    history_len = 0
    
    depth = 9
    
    # 関数呼び出し引数を修正
    score, f, t, is_drop = ai.alpha_beta_jit(
        board, hand, depth, player, -999999, 999999, current_hash, TT_TABLE, history_buffer, history_len
    )

    if f == -1:
        return board, hand, score, "Resign", "win" # 投了

    move_tuple = (f, t, is_drop)
    ai.make_move_jit(board, hand, move_tuple, player, current_hash)

    move_desc = f"{f}->{t}" if is_drop == 0 else f"Drop {f}->{t}"

    win_result = ai.win_check_jit(board, hand, player)
    status = 'playing'
    if win_result == ai.WIN_SCORE:
        status = 'win'

    return board, hand, score, move_desc, status

if __name__ == '__main__':
    app.run(debug=True, port=5000)
from flask import Flask, render_template, request, jsonify
import numpy as np
import game_logic as ai
from numba import njit

app = Flask(__name__)

# JITコンパイルの初期化 (サーバー起動時に一度だけ走らせる)
print("Initializing AI & JIT compilation...")
TT_TABLE = np.zeros((ai.TT_SIZE, 7), dtype=np.int64)
_b = np.zeros(12, dtype=np.int8)
_h = np.zeros((2, 6), dtype=np.int8)
_hash = ai.compute_hash(_b, _h, 1)
# ダミー実行でコンパイルを済ませる
ai.get_legal_moves_jit(_b, _h, 1)
ai.alpha_beta_jit(_b, _h, 1, 1, -10000, 10000, _hash, TT_TABLE, np.array([], dtype=np.int64))
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
    """ゲーム開始。人間が後手(Player2)なら、AI(Player1)が初手を指して返す"""
    data = request.json
    human_player = int(data.get('human_player', 1)) # 1 or 2
    
    board = np.array(INITIAL_BOARD, dtype=np.int8)
    hand = np.zeros((2, 6), dtype=np.int8)
    
    # 人間が先手ならそのまま盤面を返す
    if human_player == 1:
        return jsonify({
            'status': 'playing',
            'board': board.tolist(),
            'hand': hand.tolist(),
            'turn': 1 # Player 1's turn
        })
    
    # 人間が後手なら、AI (Player 1) が初手を指す
    else:
        ai_player = 1
        board, hand, score, move_desc, game_status = run_ai_turn(board, hand, ai_player)
        
        return jsonify({
            'status': game_status, # playing, win, lose
            'board': board.tolist(),
            'hand': hand.tolist(),
            'turn': 2, # Next is Player 2 (Human)
            'message': f"AI Move: {move_desc}"
        })

@app.route('/move', methods=['POST'])
def make_move():
    """
    人間の指し手を受け取り、検証し、反映し、AIが指し返し、最終結果を返す
    """
    data = request.json
    
    # 受信データの復元
    board = np.array(data['board'], dtype=np.int8)
    hand = np.array(data['hand'], dtype=np.int8)
    human_player = int(data['human_player'])
    ai_player = 2 if human_player == 1 else 1
    
    move_f = int(data['move']['from'])
    move_t = int(data['move']['to'])
    move_d = int(data['move']['drop'])

    # 1. 人間の手の正当性チェック (Illegal Move Check)
    if not is_legal_move(board, hand, human_player, move_f, move_t, move_d):
        return jsonify({'status': 'error', 'message': 'その手は反則です (Illegal Move)'})

    # 2. 人間の手を反映
    current_hash = ai.compute_hash(board, hand, human_player)
    move_tuple = (move_f, move_t, move_d)
    captured, is_promoted, _ = ai.make_move_jit(board, hand, move_tuple, human_player, current_hash)

    # 3. 勝敗判定 (人間が勝ったか？)
    win_result = ai.win_check_jit(board, hand, human_player)
    if win_result == ai.WIN_SCORE:
        return jsonify({
            'status': 'win', 
            'board': board.tolist(), 
            'hand': hand.tolist(),
            'message': 'あなたの勝ちです！ (You Win!)'
        })

    # 4. AIのターン
    board, hand, score, move_desc, ai_status = run_ai_turn(board, hand, ai_player)

    if ai_status == 'win':
        # AI視点でWin = 人間の負け
        return jsonify({
            'status': 'lose',
            'board': board.tolist(),
            'hand': hand.tolist(),
            'message': 'あなたの負けです... (You Lose)'
        })
    
    # 継続
    return jsonify({
        'status': 'playing',
        'board': board.tolist(),
        'hand': hand.tolist(),
        'score': int(score),
        'message': f"AI: {move_desc}"
    })


def is_legal_move(board, hand, player, f, t, d):
    """jitの合法手生成を使って、手がリストにあるか確認する"""
    moves = ai.get_legal_moves_jit(board, hand, player)
    for i in range(len(moves)):
        # moves[i] = [from, to, is_drop]
        if moves[i][0] == f and moves[i][1] == t and moves[i][2] == d:
            return True
    return False

def run_ai_turn(board, hand, player):
    """AIに思考させ、手を反映して返す"""
    current_hash = ai.compute_hash(board, hand, player)
    dummy_hist = np.array([], dtype=np.int64) # 千日手チェックは簡易化のため省略
    depth = 10
    
    score, f, t, is_drop = ai.alpha_beta_jit(
        board, hand, depth, player, -999999, 999999, current_hash, TT_TABLE, dummy_hist
    )

    # 投了判定
    if f == -1:
        return board, hand, score, "Resign", "win" # AI投了 = 相手(人間)の勝ち

    # 手の反映
    move_tuple = (f, t, is_drop)
    ai.make_move_jit(board, hand, move_tuple, player, current_hash)

    # 手の文字列表現 (デバッグ用)
    move_desc = f"{f}->{t}" if is_drop == 0 else f"Drop {f}->{t}"

    # AIが勝ったかチェック
    win_result = ai.win_check_jit(board, hand, player)
    status = 'playing'
    if win_result == ai.WIN_SCORE:
        status = 'win' # AI視点の勝ち

    return board, hand, score, move_desc, status

if __name__ == '__main__':
    app.run(debug=True, port=5000)
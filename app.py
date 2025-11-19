from flask import Flask, render_template, request, jsonify
import numpy as np
import game_logic as ai  # あなたのコードをインポート
from numba import njit

app = Flask(__name__)

# グローバル変数としてTTテーブルなどを保持
TT_TABLE = np.zeros((ai.TT_SIZE, 7), dtype=np.int64)
HISTORY = []  # 簡易的な履歴（本番ではセッション管理推奨）

# サーバー起動時に一度JITコンパイルを済ませておく（初回遅延防止）
print("Initializing AI & JIT compilation...")
_b = np.zeros(12, dtype=np.int8)
_h = np.zeros((2, 6), dtype=np.int8)
_hash = ai.compute_hash(_b, _h, 1)
ai.alpha_beta_jit(_b, _h, 1, 1, -10000, 10000, _hash, TT_TABLE, np.array([], dtype=np.int64))
print("Ready!")

@app.route('/')
def index():
    # 初期盤面の定義 (空の盤面や初期配置)
    # ここでは初期配置をセットします
    # ライオン:1, キリン:2, ゾウ:3, ヒヨコ:4, ニワトリ:5 (負の値は相手)
    # 配列順: A1, B1, C1, A2, B2, C2...
    initial_board = [
        -2, -1, -3, # 相手 G L E
         0, -4,  0, # 相手 . C .
         0,  4,  0, # 自分 . C .
         3,  1,  2  # 自分 E L G
    ]
    return render_template('index.html', initial_board=initial_board)

@app.route('/ai_move', methods=['POST'])
def ai_move():
    data = request.json
    
    # JSから送られてきたデータをNumpy配列(int8)に変換
    board = np.array(data['board'], dtype=np.int8)
    hand = np.array(data['hand'], dtype=np.int8) # shape (2, 6)
    player = 2 # AIは常にPlayer 2 (後手/上側) と仮定
    
    current_hash = ai.compute_hash(board, hand, player)
    
    # AI思考 (深さは調整可)
    depth = 10
    # 履歴は今回は簡易的に空で渡します（千日手対応するなら履歴をJSから送る必要あり）
    dummy_hist = np.array([], dtype=np.int64) 
    
    score, f, t, is_drop = ai.alpha_beta_jit(
        board, hand, depth, player, -999999, 999999, current_hash, TT_TABLE, dummy_hist
    )
    
    if f == -1:
        return jsonify({'status': 'game_over', 'message': 'AI投了'})

    # AIの手を適用して新しい盤面を返す
    captured, is_promoted, _ = ai.make_move_jit(board, hand, (f, t, is_drop), player, current_hash)
    
    return jsonify({
        'status': 'ok',
        'move': {'from': int(f), 'to': int(t), 'drop': int(is_drop)},
        'board': board.tolist(),
        'hand': hand.tolist(),
        'score': int(score)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
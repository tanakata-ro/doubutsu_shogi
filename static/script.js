// 盤面状態 (0:空, 正:自分, 負:敵)
// 初期配置
let board = [
    -2, -1, -3,
     0, -4,  0,
     0,  4,  0,
     3,  1,  2
];

// 持ち駒 (idx 0:未使用, 1:L, 2:G, 3:E, 4:C, 5:H)
// hand[0] = 自分の持ち駒, hand[1] = 敵の持ち駒
let hand = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
];

// 駒の文字定義 (著作権フリーな文字で表現)
const PIECE_CHARS = {
    1: "王", // ライオン (LION)
    2: "麒", // キリン (GIRAFFE)
    3: "象", // ゾウ (ELEPHANT)
    4: "歩", // ヒヨコ (CHICK)
    5: "金"  // ニワトリ (CHICKEN)
};

let selectedIndex = -1; // -1:未選択, 0-11:盤面, 100+:持ち駒
let isPlayerTurn = true;

function render() {
    const boardEl = document.getElementById('board');
    boardEl.innerHTML = '';

    // 盤面描画
    board.forEach((piece, idx) => {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.dataset.index = idx;
        
        if (piece !== 0) {
            const type = Math.abs(piece);
            cell.innerText = PIECE_CHARS[type] || "?";
            cell.classList.add(piece > 0 ? 'piece-own' : 'piece-enemy');
            if (type === 5) cell.classList.add('promoted'); // ニワトリ強調
        }
        
        if (idx === selectedIndex) cell.classList.add('selected');
        
        cell.onclick = () => handleClick(idx);
        boardEl.appendChild(cell);
    });

    // 持ち駒描画
    renderHand(0, 'hand-1'); // 自分
    renderHand(1, 'hand-2'); // 敵
}

function renderHand(playerIdx, elementId) {
    const container = document.getElementById(elementId);
    container.innerHTML = '';
    
    // 1(L)は持ち駒にならないので2(G)から
    for(let type=2; type<=5; type++) {
        const count = hand[playerIdx][type];
        if(count > 0) {
            const p = document.createElement('div');
            p.className = 'hand-piece';
            p.innerHTML = PIECE_CHARS[type] + `<span>x${count}</span>`;
            if (playerIdx === 1) p.classList.add('piece-enemy'); // 敵の持ち駒
            else p.classList.add('piece-own');

            // 自分の持ち駒だけクリック可能
            if (playerIdx === 0) {
                // 持ち駒選択時は 100 + type とする
                if (selectedIndex === 100 + type) p.classList.add('selected');
                p.onclick = () => handleHandClick(type);
            }
            container.appendChild(p);
        }
    }
}

function handleHandClick(type) {
    if(!isPlayerTurn) return;
    if (selectedIndex === 100 + type) {
        selectedIndex = -1; // 解除
    } else {
        selectedIndex = 100 + type;
    }
    render();
}

function handleClick(idx) {
    if(!isPlayerTurn) return;

    const clickedPiece = board[idx];
    const isMyPiece = clickedPiece > 0;

    // 1. 持ち駒を選択中の場合 -> 打つ
    if (selectedIndex >= 100) {
        const type = selectedIndex - 100;
        if (clickedPiece === 0) {
            // 空きますなら打てる（詳細なルール判定はサーバーでもやるが、簡易チェック）
            executeMove(type, idx, true);
        }
        selectedIndex = -1;
        render();
        return;
    }

    // 2. 盤面の駒を選択中 -> 移動
    if (selectedIndex !== -1) {
        // 同じマスをクリック -> 解除
        if (selectedIndex === idx) {
            selectedIndex = -1;
        } 
        // 自分の駒をクリック -> 選択し直し
        else if (isMyPiece) {
            selectedIndex = idx;
        }
        // 移動実行 (空きます or 敵の駒)
        else {
            executeMove(selectedIndex, idx, false);
            selectedIndex = -1;
        }
    } 
    // 3. 何も選択していない -> 自分の駒なら選択
    else if (isMyPiece) {
        selectedIndex = idx;
    }

    render();
}

async function executeMove(from, to, isDrop) {
    // 暫定的にJS側で盤面更新（アニメーション用）
    // ※ 本来はサーバーの正規判定を待つべきだが、レスポンス向上のため
    // 簡易実装: サーバーに今の状態を送って、AIに打たせる
    
    // 自分の手をサーバーへ送信する前に、JS上で仮反映させるロジックは複雑なので
    // ここでは「サーバーに手だけ送る」のではなく
    // 「自分の手を反映した盤面を作ってAIに投げる」方式にします。
    // ただし、njit_doubtsuのロジックがサーバーにあるので、
    // 今回は「自分の手を適用するAPI」を作らず、簡易的にJSで手を反映してしまいます。
    
    // === JSでの簡易更新 (本来はここでルールチェックが必要) ===
    if (isDrop) {
        board[to] = from; // 自分の駒として配置
        hand[0][from]--;
    } else {
        const piece = board[from];
        const target = board[to];
        
        // 捕獲
        if (target < 0) {
            let capType = Math.abs(target);
            if (capType === 5) capType = 4; // ニワトリはヒヨコに戻る
            hand[0][capType]++;
        }
        
        // 成り判定 (ヒヨコが最奥へ)
        if (Math.abs(piece) === 4 && Math.floor(to / 3) === 0) {
            board[to] = 5; // ニワトリ
        } else {
            board[to] = piece;
        }
        board[from] = 0;
    }
    
    isPlayerTurn = false;
    document.getElementById('status').innerText = "Thinking...";
    render();

    // サーバーへ送信
    try {
        const response = await fetch('/ai_move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                board: board,
                hand: hand
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'ok') {
            // サーバーから返ってきた「AIが打った後の盤面」で更新
            // ※ data.board はAIの手番完了後の状態
            board = data.board;
            hand = data.hand;
            
            document.getElementById('status').innerText = `Eval: ${data.score}`;
        } else {
            alert(data.message);
        }
    } catch (e) {
        console.error(e);
        alert("通信エラー");
    }
    
    isPlayerTurn = true;
    render();
}

// 初期描画
render();
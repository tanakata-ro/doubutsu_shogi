let board = [];
let hand = [[0]*6, [0]*6];
let humanPlayerId = 1; 
let isPlayerTurn = false;
let selectedIndex = -1; // -1:None, 0-11:Board, 100+:Hand

// デザイン変更: ひらがな
const PIECE_CHARS = {
    1: "ら", // ライオン
    2: "き", // キリン
    3: "ぞ", // ゾウ
    4: "ひ", // ヒヨコ
    5: "に"  // ニワトリ
};

// --- ゲーム開始処理 ---
async function startGame(pid) {
    humanPlayerId = pid;
    document.getElementById('setup-area').style.display = 'none';
    document.getElementById('game-area').style.display = 'block';
    
    const boardEl = document.getElementById('board');
    if (humanPlayerId === 2) {
        boardEl.classList.add('view-p2');
    } else {
        boardEl.classList.remove('view-p2');
    }

    document.getElementById('status').style.display = 'block';
    document.getElementById('message').innerText = "Initializing...";

    try {
        const res = await fetch('/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ human_player: humanPlayerId })
        });
        const data = await res.json();
        updateGameState(data);

    } catch (e) {
        console.error(e);
        alert("Connection Error");
    }
}

function updateGameState(data) {
    board = data.board;
    hand = data.hand;
    
    // ターン切り替え時に選択を解除
    selectedIndex = -1;

    if (data.status === 'win') {
        render(); // 最後の盤面を描画
        endGame(true, data.message);
    } else if (data.status === 'lose') {
        render();
        endGame(false, data.message);
    } else if (data.status === 'error') {
        alert(data.message);
        isPlayerTurn = true; 
        render();
    } else {
        isPlayerTurn = true;
        document.getElementById('status').innerText = "Your Turn";
        if (data.message) document.getElementById('message').innerText = data.message;
        render();
    }
}

function endGame(isWin, msg) {
    isPlayerTurn = false;
    document.getElementById('status').innerText = isWin ? "WINNER" : "GAME OVER";
    document.getElementById('status').style.color = isWin ? "#4ecdc4" : "#ff6b6b";
    document.getElementById('message').innerText = msg;
}

// --- 描画処理 ---
function render() {
    const boardEl = document.getElementById('board');
    boardEl.innerHTML = '';

    // ■ 有効な移動先を計算
    const validMoves = getValidMoves(selectedIndex);

    // 盤面
    board.forEach((piece, idx) => {
        const cell = document.createElement('div');
        cell.className = 'cell';
        
        if (piece !== 0) {
            const type = Math.abs(piece);
            const span = document.createElement('span');
            span.innerText = PIECE_CHARS[type];
            
            const owner = piece > 0 ? 1 : 2;
            if (owner === humanPlayerId) {
                span.classList.add('piece-own');
            } else {
                span.classList.add('piece-enemy');
            }
            if (type === 5) span.classList.add('promoted');
            cell.appendChild(span);
        }
        
        // 選択中スタイル
        if (idx === selectedIndex) cell.classList.add('selected');

        // ★ガイド表示: 有効な移動先ならハイライト
        if (validMoves.includes(idx)) {
            cell.classList.add('suggested');
        }
        
        cell.onclick = () => handleBoardClick(idx, validMoves);
        boardEl.appendChild(cell);
    });

    // 持ち駒
    const myHandIdx = humanPlayerId - 1;
    const enemyHandIdx = (humanPlayerId === 1) ? 1 : 0;
    renderHand(myHandIdx, 'hand-self', true);
    renderHand(enemyHandIdx, 'hand-enemy', false);
}

function renderHand(playerIdx, elementId, isSelf) {
    const container = document.getElementById(elementId);
    container.innerHTML = '';
    
    for(let type=2; type<=5; type++) {
        const count = hand[playerIdx][type];
        if(count > 0) {
            const p = document.createElement('div');
            p.className = 'hand-piece';
            p.innerHTML = PIECE_CHARS[type] + `<span>x${count}</span>`;
            p.classList.add(isSelf ? 'piece-own' : 'piece-enemy');
            
            if (isSelf) {
                if (selectedIndex === 100 + type) p.classList.add('selected');
                p.onclick = () => handleHandClick(type);
            }
            container.appendChild(p);
        }
    }
}

// --- ロジック: 有効な移動先を計算する ---
function getValidMoves(selIdx) {
    if (selIdx === -1) return [];
    
    const moves = [];
    const myDir = (humanPlayerId === 1) ? -1 : 1; // P1はyが減る方向、P2はyが増える方向

    // 1. 持ち駒を選択中 (Drop)
    if (selIdx >= 100) {
        // 空いているマスならどこでもOK
        for(let i=0; i<12; i++) {
            if (board[i] === 0) moves.push(i);
        }
        return moves;
    }

    // 2. 盤上の駒を選択中 (Move)
    const idx = selIdx;
    const piece = board[idx];
    if (piece === 0) return []; // 空マス選択時はなし

    const type = Math.abs(piece);
    const cx = idx % 3;
    const cy = Math.floor(idx / 3);

    // 移動パターンの定義 (相対座標 [dx, dy])
    // ※ myDir を掛けて向きを補正する
    let offsets = [];

    if (type === 1) { // ライオン (全方向)
        offsets = [[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];
    } else if (type === 2) { // キリン (縦横)
        offsets = [[0,-1],[0,1],[-1,0],[1,0]];
    } else if (type === 3) { // ゾウ (斜め)
        offsets = [[-1,-1],[1,-1],[-1,1],[1,1]];
    } else if (type === 4) { // ヒヨコ (前のみ)
        offsets = [[0, 1 * myDir]];
    } else if (type === 5) { // ニワトリ (金: 前横後なし)
        // 基準は「前」が myDir の方向
        // 前, 前左, 前右, 横左, 横右, 後ろ
        offsets = [
            [0, 1*myDir], [-1, 1*myDir], [1, 1*myDir], // 前3方向
            [-1, 0], [1, 0], // 横
            [0, -1*myDir] // 後ろ
        ];
    }

    // 盤外チェックと味方駒チェック
    offsets.forEach(off => {
        const tx = cx + off[0];
        const ty = cy + off[1];

        if (tx >= 0 && tx < 3 && ty >= 0 && ty < 4) {
            const tIdx = tx + ty * 3;
            const target = board[tIdx];
            
            // 空きます or 敵の駒ならOK
            // 自分の駒(humanPlayerIdと同じ符号)がいるならNG
            // boardの値: P1は正、P2は負
            const isMyPiece = (humanPlayerId === 1 && target > 0) || (humanPlayerId === 2 && target < 0);
            
            if (!isMyPiece) {
                moves.push(tIdx);
            }
        }
    });

    return moves;
}


// --- 操作イベント ---

function handleHandClick(type) {
    if (!isPlayerTurn) return;
    selectedIndex = (selectedIndex === 100 + type) ? -1 : 100 + type;
    render();
}

function handleBoardClick(idx, validMoves) {
    if (!isPlayerTurn) return;

    const piece = board[idx];
    // 自分の駒かどうか
    const isMyPiece = (humanPlayerId === 1 && piece > 0) || (humanPlayerId === 2 && piece < 0);

    // 1. 何かを選択している時
    if (selectedIndex !== -1) {
        // A. 同じ場所 -> 解除
        if (selectedIndex === idx) {
            selectedIndex = -1;
            render();
            return;
        }
        
        // B. 有効な移動先をクリック -> 移動実行
        if (validMoves.includes(idx)) {
            const isDrop = (selectedIndex >= 100);
            const from = isDrop ? (selectedIndex - 100) : selectedIndex;
            sendMove(from, idx, isDrop ? 1 : 0);
            return;
        }

        // C. 自分の別の駒をクリック -> 選択切り替え
        if (isMyPiece) {
            selectedIndex = idx;
            render();
            return;
        }

        // D. それ以外 (無効な場所や遠くの敵) -> ★何もしない (無視)
        return;
    } 
    
    // 2. 何も選択していない時 -> 自分の駒なら選択
    if (isMyPiece) {
        selectedIndex = idx;
        render();
    }
}

// --- 送信処理 (Optimistic UI) ---
async function sendMove(from, to, isDrop) {
    // 即座にロック
    const savedSelectedIndex = selectedIndex;
    selectedIndex = -1;
    isPlayerTurn = false;

    const currentBoard = [...board];
    const currentHand = [ [...hand[0]], [...hand[1]] ];

    // 画面反映
    simulateMove(from, to, isDrop);
    document.getElementById('status').innerText = "Thinking...";
    render(); 

    try {
        const res = await fetch('/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                board: currentBoard, 
                hand: currentHand,   
                human_player: humanPlayerId,
                move: { from: from, to: to, drop: isDrop }
            })
        });
        const data = await res.json();
        updateGameState(data);

    } catch (e) {
        console.error(e);
        alert("Communication Error");
        board = currentBoard;
        hand = currentHand;
        isPlayerTurn = true;
        render();
    }
}

function simulateMove(from, to, isDrop) {
    const myIdx = humanPlayerId - 1;
    
    if (isDrop) {
        board[to] = (humanPlayerId === 1) ? from : -from;
        hand[myIdx][from]--;
    } else {
        const piece = board[from];
        const target = board[to];

        if (target !== 0) {
            let capType = Math.abs(target);
            if (capType === 5) capType = 4; 
            hand[myIdx][capType]++;
        }

        const type = Math.abs(piece);
        const row = Math.floor(to / 3);
        let isPromote = false;
        
        if (type === 4) {
            if (humanPlayerId === 1 && row === 0) isPromote = true;
            if (humanPlayerId === 2 && row === 3) isPromote = true;
        }

        if (isPromote) {
            board[to] = (humanPlayerId === 1) ? 5 : -5;
        } else {
            board[to] = piece;
        }
        board[from] = 0;
    }
}
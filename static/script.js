let board = [];
let hand = [[0]*6, [0]*6];
let humanPlayerId = 1; // 1 or 2
let isPlayerTurn = false;
let selectedIndex = -1; // -1:None, 0-11:Board, 100+:Hand

const PIECE_CHARS = {
    1: "王", 2: "麒", 3: "象", 4: "歩", 5: "金"
};

// ゲーム開始処理
async function startGame(pid) {
    humanPlayerId = pid;
    document.getElementById('setup-area').style.display = 'none';
    document.getElementById('game-area').style.display = 'block';
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

// サーバーからのレスポンスで全状態を更新
function updateGameState(data) {
    board = data.board;
    hand = data.hand;
    
    render();

    if (data.status === 'win') {
        endGame(true, data.message);
    } else if (data.status === 'lose') {
        endGame(false, data.message);
    } else if (data.status === 'error') {
        // 手を戻すなどの処理が必要だが、今回はAlertのみ
        alert(data.message);
        isPlayerTurn = true; 
    } else {
        // ゲーム継続
        isPlayerTurn = true;
        document.getElementById('status').innerText = "Your Turn";
        if (data.message) document.getElementById('message').innerText = data.message;
    }
}

function endGame(isWin, msg) {
    isPlayerTurn = false;
    document.getElementById('status').innerText = isWin ? "WINNER" : "GAME OVER";
    document.getElementById('status').style.color = isWin ? "#4ecdc4" : "#ff6b6b";
    document.getElementById('message').innerText = msg;
    alert(msg);
}

// 描画
function render() {
    const boardEl = document.getElementById('board');
    boardEl.innerHTML = '';

    // 盤面
    board.forEach((piece, idx) => {
        const cell = document.createElement('div');
        cell.className = 'cell';
        
        if (piece !== 0) {
            const type = Math.abs(piece);
            cell.innerText = PIECE_CHARS[type];
            
            // 駒の持ち主判定 (値が正ならPlayer1, 負ならPlayer2)
            const owner = piece > 0 ? 1 : 2;
            
            // 自分(humanPlayerId)の駒かどうか
            if (owner === humanPlayerId) {
                cell.classList.add('piece-own');
            } else {
                cell.classList.add('piece-enemy');
            }
            if (type === 5) cell.classList.add('promoted');
        }
        
        if (idx === selectedIndex) cell.classList.add('selected');
        cell.onclick = () => handleBoardClick(idx);
        boardEl.appendChild(cell);
    });

    // 持ち駒
    // hand配列: [0]がPlayer1, [1]がPlayer2
    // しかし画面下(自分)には humanPlayerId の持ち駒を表示したい
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

// クリック処理
function handleHandClick(type) {
    if (!isPlayerTurn) return;
    selectedIndex = (selectedIndex === 100 + type) ? -1 : 100 + type;
    render();
}

function handleBoardClick(idx) {
    if (!isPlayerTurn) return;

    const piece = board[idx];
    const owner = piece > 0 ? 1 : 2;
    const isMyPiece = (piece !== 0 && owner === humanPlayerId);

    // 1. 持ち駒を選択中 -> 打つ
    if (selectedIndex >= 100) {
        const type = selectedIndex - 100;
        if (piece === 0) {
            sendMove(type, idx, 1); // Drop
        } else {
            // 打てない場所（駒がある）
            selectedIndex = -1;
            render();
        }
        return;
    }

    // 2. 盤上の駒を選択中 -> 移動
    if (selectedIndex !== -1) {
        if (selectedIndex === idx) {
            selectedIndex = -1; // 解除
        } else if (isMyPiece) {
            selectedIndex = idx; // 変更
        } else {
            // 移動実行 (空きます or 敵)
            sendMove(selectedIndex, idx, 0); // Move
        }
    } 
    // 3. 未選択 -> 自分の駒を選択
    else if (isMyPiece) {
        selectedIndex = idx;
    }
    render();
}

async function sendMove(from, to, isDrop) {
    selectedIndex = -1;
    isPlayerTurn = false;
    document.getElementById('status').innerText = "Thinking...";
    document.getElementById('message').innerText = "";
    render(); // 選択解除を反映

    try {
        const res = await fetch('/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                board: board,
                hand: hand,
                human_player: humanPlayerId,
                move: { from: from, to: to, drop: isDrop }
            })
        });
        const data = await res.json();
        updateGameState(data);

    } catch (e) {
        console.error(e);
        alert("Communication Error");
        isPlayerTurn = true;
    }
}
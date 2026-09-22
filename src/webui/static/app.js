'use strict';

const $ = (id) => document.getElementById(id);

const state = {
  img: null,          // 去畸变后的源图
  quad: null,         // [[x,y] x4]，图像坐标系
  quadGuess: null,    // 载入时服务端给的起点，供"复位四点"用
  committedQuad: null,// 上次成功导出的四点，供"载入上次结果"用
  linePoints: null,   // [[p1,p2] x4]，交点模式下的四条线（TOP/BOTTOM/LEFT/RIGHT）
  editMode: 'corner', // 'corner' = 拖角点；'line' = 拖四条线，角点由交点实时算出
  view: { scale: 1, tx: 0, ty: 0 },
  drag: null,         // { mode:'handle'|'pan'|'line-end', i, ei, ox, oy, tx0, ty0 }
  selected: -1,
  previewTimer: null,
  previewing: false,
  draggingScale: false,
  lastPreview: null,
  fit: null,          // 最近一次标定的逐帧数据，给柱状图用
};

const LABELS = ['TL', 'TR', 'BL', 'BR'];
const LINE_NAMES = ['TOP', 'BOTTOM', 'LEFT', 'RIGHT'];
const HANDLE_HIT_PX = 14;

// 放大镜固定 5x。放大窗钉在源图区域右上角，中心锁定"当前选中的点"，
// 不跟随鼠标——微调点位时视线不必在光标与放大镜之间来回跳。
const LOUPE_SCALE = 5;
const LOUPE_SIZE = 160;   // 放大窗边长，与 style.css 里 #c-loupe 的尺寸一致

/** 把一行直线 a1->a2 和 b1->b2 求交点；接近平行时返回 null。 */
function intersectLines(a1, a2, b1, b2) {
  const [x1, y1] = a1, [x2, y2] = a2, [x3, y3] = b1, [x4, y4] = b2;
  const den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(den) < 1e-9) return null;
  const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
  return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)];
}

/** 由四条线（TOP/BOTTOM/LEFT/RIGHT）算角点；任何一对平行 → null。 */
function quadFromLines(linePoints) {
  const [top, bottom, left, right] = linePoints;
  const tl = intersectLines(top[0], top[1], left[0], left[1]);
  const tr = intersectLines(top[0], top[1], right[0], right[1]);
  const bl = intersectLines(bottom[0], bottom[1], left[0], left[1]);
  const br = intersectLines(bottom[0], bottom[1], right[0], right[1]);
  if (!tl || !tr || !bl || !br) return null;
  if (![tl, tr, bl, br].every((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]))) {
    return null;
  }
  return [tl, tr, bl, br];
}

/** 由角点反推四条边，落到原图坐标系。 */
function linesFromQuad(quad) {
  const [tl, tr, bl, br] = quad.map((p) => p.slice());
  return [
    [tl, tr],   // TOP
    [bl, br],   // BOTTOM
    [tl, bl],   // LEFT
    [tr, br],   // RIGHT
  ];
}

/** 渲染重投影误差柱状图，并按给定阈值实时标注哪些会被剔除。

    fit.all_names / fit.all_errors 是**剔除之前全体视图**的误差，所以换任何阈值都能
    立刻数出会删多少张，不必等重新标定跑完。老版本 fit 数据没有这两个字段时，
    退回用 per_view + dropped 拼一份（此时只能反映"上一次实际发生的结果"）。
 */
function drawErrorChart(fit, threshold) {
  const cv = $('c-reproj');
  const cssW = cv.clientWidth || 560;
  const cssH = 140;
  const dpr = window.devicePixelRatio || 1;
  cv.width = cssW * dpr;
  cv.height = cssH * dpr;
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  // 还没有标定数据时给个明确的空态，别留一块莫名其妙的白板
  if (!fit) {
    ctx.fillStyle = '#b4b2a9';
    ctx.font = '12px -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('还没有标定数据', cssW / 2, cssH / 2 - 10);
    ctx.fillText('导入素材后会自动标定一次，也可以直接点下面的「运行标定」',
      cssW / 2, cssH / 2 + 12);
    const cnt0 = $('calib-count');
    if (cnt0) {
      cnt0.textContent = '尚无 calib.json —— 先跑一次标定';
      cnt0.className = 'hint mono';
    }
    return;
  }

  let bars = [];
  if (fit.all_names && fit.all_names.length) {
    bars = fit.all_names.map((name, i) => ({ name, error: fit.all_errors[i] }));
  } else {
    bars = (fit.per_view || []).map((e, i) => ({ name: fit.used_names[i], error: e }))
      .concat((fit.dropped || []).map((d) => ({ name: d.name, error: d.error })));
  }
  if (!bars.length) return;
  bars.sort((a, b) => a.name.localeCompare(b.name));

  const hasThr = typeof threshold === 'number' && isFinite(threshold) && threshold > 0;
  const cut = hasThr ? bars.filter((b) => b.error > threshold).length : 0;

  // 数量行
  const cnt = $('calib-count');
  if (cnt) {
    const applied = fit.threshold;
    const pending = hasThr && (applied == null || Math.abs(applied - threshold) > 1e-9);
    cnt.textContent = hasThr
      ? `将剔除 ${cut} / 共 ${bars.length} 张（阈值 ${threshold}）`
        + (pending ? ' · 预览，点「运行标定」生效' : '')
      : `不剔除 · 共 ${bars.length} 张`;
    if (fit.rms) cnt.textContent += `  ·  上次 RMS ${fit.rms.toFixed(3)} px`;
    cnt.className = 'hint mono' + (cut ? ' bad' : '');
  }

  // 画布
  const maxErr = Math.max(...bars.map((b) => b.error), hasThr ? threshold : 0) * 1.1;
  const padL = 34, padR = 8, padT = 8, padB = 14;
  const plotW = cssW - padL - padR;
  const plotH = cssH - padT - padB;
  const slot = plotW / bars.length;
  const barW = Math.max(2, slot - 1.5);

  ctx.font = '10px ui-monospace, monospace';
  ctx.strokeStyle = 'rgba(0,0,0,.08)';
  ctx.lineWidth = 1;
  ctx.textAlign = 'right';
  ctx.fillStyle = '#96958f';
  for (let i = 0; i <= 4; i++) {
    const y = padT + plotH * (1 - i / 4);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(cssW - padR, y); ctx.stroke();
    ctx.fillText((maxErr * i / 4).toFixed(2), padL - 4, y + 3);
  }

  // 柱：超过阈值红、其余绿
  bars.forEach((b, i) => {
    const x = padL + i * slot + (slot - barW) / 2;
    const h = Math.max(1, (b.error / maxErr) * plotH);
    ctx.fillStyle = (hasThr && b.error > threshold) ? '#a32d2d' : '#639922';
    ctx.fillRect(x, padT + plotH - h, barW, h);
  });

  // 阈值虚线
  if (hasThr) {
    const yT = padT + plotH * (1 - threshold / maxErr);
    ctx.strokeStyle = 'rgba(163,45,45,.75)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(padL, yT); ctx.lineTo(cssW - padR, yT);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.textAlign = 'left';
    ctx.fillStyle = '#a32d2d';
    ctx.fillText(`${threshold}`, padL + 4, yT - 3);
  }
}

// ---------------------------------------------------------------- 基础

function log(text, cls) {
  const el = $('log');
  const span = document.createElement('span');
  if (cls) span.className = cls;
  span.textContent = String(text).replace(/\s+$/, '') + '\n';
  el.appendChild(span);
  el.scrollTop = el.scrollHeight;
}

function clearLog() { $('log').textContent = ''; }

async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
  const res = await fetch(path, opt);
  let data;
  try {
    data = await res.json();
  } catch (e) {
    throw new Error('服务端返回了非 JSON 响应（HTTP ' + res.status + '）');
  }
  if (data.error) throw new Error(data.error);
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return data;
}

async function upload(path, formData) {
  const res = await fetch(path, { method: 'POST', body: formData });
  let data;
  try {
    data = await res.json();
  } catch (e) {
    throw new Error('服务端返回了非 JSON 响应（HTTP ' + res.status + '）');
  }
  if (data.error) throw new Error(data.error);
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return data;
}

function setChip(el, text, cls) {
  el.textContent = text;
  el.className = 'chip' + (cls ? ' ' + cls : '');
}

function kv(container, rows) {
  container.innerHTML = '';
  rows.forEach(([k, v, cls]) => {
    const d = document.createElement('div');
    const a = document.createElement('span');
    a.textContent = k;
    const b = document.createElement('span');
    b.textContent = v;
    if (cls) b.className = cls;
    d.append(a, b);
    container.appendChild(d);
  });
}

/** 把耗时按钮包起来，执行期间禁用，避免重复点击。

    用 innerHTML 而不是 textContent 存取原文案：有些按钮里还嵌着子元素
    （比如画廊按钮里的张数徽标），textContent 会把它们一并抹掉。
 */
async function withBusy(btn, fn) {
  const label = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = '运行中…';
  try {
    return await fn();
  } finally {
    btn.disabled = false;
    btn.innerHTML = label;
  }
}

// ---------------------------------------------------------------- 状态

// 素材库的分拣依据与当前规格对不上（或压根没记过）：在「标定板规格」和「导入素材」
// 两张卡片上常驻显示。只往日志里打一行的话，刷新一次页面警告就没了，而脏状态还原样
// 留着 —— 用户照样会在逆透视候选里看到混进来的棋盘照。
function renderMaterialWarning(reason) {
  ['board-stale', 'import-stale'].forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.textContent = reason ? '⚠ ' + reason : '';
    el.hidden = !reason;
  });
}

// 素材导入事务的残留（/api/status 的 material_transaction）。分两档：
//   unsafe（project.json 里还有 pending，或正式目录旁边留着 .old）
//     → 正式素材库处于不可判定状态，后端连相机标定 / 逆透视选图 / 增量导入都拒绝，
//       红色常驻。
//   只剩 *.staging → 旧库完好，可以继续用，只是下一次「覆盖整个素材库」会被拦，
//       黄色提示。
// 这里刻意只做"显示"：不提供自动恢复、也不提供一键删除残留 —— 哪一份才是用户想
// 保留的，程序猜不出来，而 --move 导入时残留里可能是原位置已经没有的唯一原件。
// 真正的拦截始终在后端，前端只是把这个事实提前告诉用户。
function renderTransactionWarning(txn) {
  const unsafe = !!(txn && txn.unsafe);
  const residue = (txn && txn.residue) || [];
  const show = !!(txn && txn.message) && (unsafe || residue.length > 0);
  ['board-txn', 'import-txn'].forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.className = unsafe ? 'stale bad' : 'stale';
    el.textContent = show ? (unsafe ? '⛔ ' : '⚠ ') + txn.message : '';
    el.hidden = !show;
  });
}

async function refreshStatus() {
  const st = await api('/api/status');
  const calib = st.calib;
  if (calib) {
    setChip($('chip-calib'), `已标定 ${calib.width}×${calib.height}`, 'ok');
    kv($('calib-info'), [
      ['fx / fy', `${calib.fx.toFixed(1)} / ${calib.fy.toFixed(1)}`],
      ['cx / cy', `${calib.cx.toFixed(1)} / ${calib.cy.toFixed(1)}`],
      ['水平视场角', `${calib.hfov.toFixed(1)}°`],
      ['畸变系数', calib.dist.map((v) => v.toFixed(4)).join(', ')],
    ]);
  } else {
    setChip($('chip-calib'), '未标定', 'warn');
    kv($('calib-info'), [['状态', '尚无 calib.json，先跑第 2 步']]);
  }
  setChip($('chip-tables'), st.tables_ready ? '查找表已生成' : '查找表未生成',
    st.tables_ready ? 'ok' : '');

  // 标定板规格：当前生效值回填到输入框，并显示"这份 calib.json 是哪块棋盘算的"
  if (st.board) {
    $('in-board-x').value = st.board.squares_x;
    $('in-board-y').value = st.board.squares_y;
    $('in-board-mm').value = st.board.square_size_mm;
    updateBoardHint();
    $('board-active').textContent = '当前生效：' + st.board.label;
  }
  if (st.calib_board) {
    const b = st.calib_board;
    $('board-active').textContent +=
      `\n本次标定使用：${b.squares_x}×${b.squares_y} 方格 / ` +
      `${b.square_size_mm} mm / 内角点 ${b.corners_x}×${b.corners_y}`;
  }
  renderMaterialWarning(st.material_stale);
  renderTransactionWarning(st.material_transaction);

  const badge = $('gallery-badge');
  if (badge) {
    badge.textContent = st.preview_count ? `（${st.preview_count} 张）` : '（暂无）';
  }

  renderCandidates(st.ipm_candidates, st.ipm_source);

  clearLog();
  log('工程根目录: ' + st.root);
  st.dirs.forEach((d) => {
    const count = !d.exists ? '缺失' : (d.files === 0 ? '空' : `${d.files} 个文件`);
    log(`  ${d.name.padEnd(14, ' ')} ${count.padEnd(10, ' ')} ${d.desc}`);
  });
  if (st.ipm_state && st.ipm_state.src_image) {
    log('\n上次逆透视标定用的原图: ' + st.ipm_state.src_image.split(/[\\/]/).pop());
  }
  log('');

  // 还没有标定数据就把柱状图切成空态，别留白板
  if (!state.fit) drawErrorChart(null, null);
  return st;
}

function renderCandidates(names, currentRel) {
  const box = $('candidates');
  box.innerHTML = '';
  if (!names.length) {
    const p = document.createElement('p');
    p.className = 'hint';
    p.textContent = 'ipm_input/ 里还没有图片，先用第 1 步导入素材。';
    box.appendChild(p);
    return;
  }
  const current = currentRel ? currentRel.split(/[\\/]/).pop() : null;
  names.forEach((name) => {
    const el = document.createElement('div');
    el.className = 'thumb' + (name === current ? ' active' : '');
    el.title = name;
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.src = '/api/image?rel=' + encodeURIComponent('ipm_input/' + name) + '&max=240';
    const span = document.createElement('span');
    span.textContent = name;
    el.append(img, span);
    el.onclick = () => loadSource(name);
    box.appendChild(el);
  });
}

async function loadSource(name) {
  const data = await api('/api/source', { name });
  const img = new Image();
  await new Promise((res, rej) => {
    img.onload = res;
    img.onerror = () => rej(new Error('预览图解码失败'));
    img.src = data.image;
  });
  state.img = img;
  state.quad = data.quad.map((p) => p.slice());
  state.quadGuess = data.quad.map((p) => p.slice());
  state.linePoints = linesFromQuad(state.quad);
  state.lineGuess = data.quad.map((p) => p.slice());
  state.selected = -1;
  $('canvas-empty').style.display = 'none';
  $('source-info').innerHTML = '';
  kv($('source-info'), [
    ['原图', data.name],
    ['尺寸', `${data.width} × ${data.height}`],
  ]);
  document.querySelectorAll('.thumbs .thumb').forEach((el) => {
    el.classList.toggle('active', el.title === data.name);
  });
  fitView();
  drawSource();
  schedulePreview(true);
  log('已载入逆透视标定原图: ' + data.name);
}

// ---------------------------------------------------------------- 画布

function canvasMetrics() {
  const wrap = $('canvas-wrap');
  const rect = wrap.getBoundingClientRect();
  return { w: Math.max(1, Math.round(rect.width)), h: Math.max(1, Math.round(rect.height)) };
}

function fitView() {
  if (!state.img) return;
  const { w, h } = canvasMetrics();
  const pad = 16;
  const s = Math.min((w - pad * 2) / state.img.width, (h - pad * 2) / state.img.height);
  state.view.scale = s > 0 ? s : 1;
  state.view.tx = (w - state.img.width * state.view.scale) / 2;
  state.view.ty = (h - state.img.height * state.view.scale) / 2;
}

function zoomTo(scale, cx, cy) {
  const v = state.view;
  const next = Math.min(40, Math.max(0.05, scale));
  // 保持光标下的图像坐标不动
  v.tx = cx - (cx - v.tx) * (next / v.scale);
  v.ty = cy - (cy - v.ty) * (next / v.scale);
  v.scale = next;
}

function drawSource() {
  // 放大窗是"当前选中点"的派生视图，和主画布同一次刷新：
  // 选中点变了、点位被拖了、图换了都会走到这里，放大窗不会漏更新。
  updateLoupe();
  const cv = $('c-source');
  const { w, h } = canvasMetrics();
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== w * dpr || cv.height !== h * dpr) {
    cv.width = w * dpr;
    cv.height = h * dpr;
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!state.img) return;

  const v = state.view;
  ctx.imageSmoothingEnabled = v.scale < 2.5;
  ctx.drawImage(state.img, v.tx, v.ty,
    state.img.width * v.scale, state.img.height * v.scale);

  if (state.editMode === 'corner') {
    if (!state.quad) return;
    drawCornerMode(ctx, state.quad.map((p) => toScreen(p, v)));
  } else {
    if (!state.linePoints) return;
    const lines = state.linePoints.map((ln) => ln.map((p) => toScreen(p, v)));
    const corners = quadFromLines(state.linePoints);
    drawLineMode(ctx, lines,
      corners ? corners.map((p) => toScreen(p, v)) : null);
  }
}

function toScreen(p, v) {
  return [p[0] * v.scale + v.tx, p[1] * v.scale + v.ty];
}

/** 四角点模式：矩形 + 4 个角点手柄。

    注意连线顺序必须是 TL→TR→BR→BL（下标 0,1,3,2）。
    角点数组本身是按 TL,TR,BL,BR 排的，直接顺序连会连出 TL-TR-BL-BR-TL，
    其中 TR-BL 和 BR-TL 正是两条对角线——画面上就是一个交叉的 X，
    看起来就像"TL 连到了 BR"。主脚本 render_preview 里用的也是 [[0,1,3,2]]，这里对齐。
 */
function drawCornerMode(ctx, pts) {
  const RECT_ORDER = [0, 1, 3, 2];

  ctx.strokeStyle = '#39d353';
  ctx.lineWidth = 2;
  ctx.beginPath();
  RECT_ORDER.forEach((idx, i) => {
    const [x, y] = pts[idx];
    if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
  });
  ctx.closePath();
  ctx.stroke();

  ctx.font = '11px ui-monospace, monospace';
  ctx.textBaseline = 'middle';
  pts.forEach(([x, y], i) => {
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fillStyle = i === state.selected ? '#ff9500' : '#e24b4a';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(0,0,0,.65)';
    ctx.strokeText(LABELS[i], x + 12, y - 12);
    ctx.fillStyle = '#fff';
    ctx.fillText(LABELS[i], x + 12, y - 12);
  });
}

/** 交点模式：4 条线 + 8 个端点 + 4 个算出来的角点（不画矩形，由线相交得出）。 */
function drawLineMode(ctx, lines, cornerPts) {
  const lineColors = ['#39d353', '#39d353', '#3781ff', '#ff9500'];

  // 四条线本体
  ctx.lineWidth = 1.5;
  lines.forEach(([a, b], i) => {
    ctx.strokeStyle = lineColors[i];
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  });

  // 8 个端点手柄
  ctx.font = '10px ui-monospace, monospace';
  ctx.textBaseline = 'middle';
  lines.forEach(([a, b], i) => {
    [a, b].forEach((p, ei) => {
      const key = `l${i}e${ei}`;
      ctx.beginPath();
      ctx.arc(p[0], p[1], 6, 0, Math.PI * 2);
      ctx.fillStyle = state.selected === key ? '#ff9500' : '#7d54ff';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(0,0,0,.65)';
      const label = `${LINE_NAMES[i][0]}${ei + 1}`;
      ctx.strokeText(label, p[0] + 10, p[1] - 10);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, p[0] + 10, p[1] - 10);
    });
  });

  // 算出来的角点：小空心圈，提示这是派生量而不是手柄
  if (cornerPts) {
    cornerPts.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }
}

function toImage(clientX, clientY) {
  const rect = $('c-source').getBoundingClientRect();
  const v = state.view;
  return [(clientX - rect.left - v.tx) / v.scale, (clientY - rect.top - v.ty) / v.scale];
}

/** 返回 { mode, i?, ei? }，mode 为 'corner' | 'line-end' | null。 */
function hitTest(clientX, clientY) {
  if (!state.img) return null;
  const rect = $('c-source').getBoundingClientRect();
  const mx = clientX - rect.left;
  const my = clientY - rect.top;
  const v = state.view;

  if (state.editMode === 'corner' && state.quad) {
    let best = -1, bestD = HANDLE_HIT_PX;
    state.quad.forEach(([x, y], i) => {
      const d = Math.hypot(x * v.scale + v.tx - mx, y * v.scale + v.ty - my);
      if (d < bestD) { bestD = d; best = i; }
    });
    return best >= 0 ? { mode: 'corner', i: best } : null;
  }

  if (state.editMode === 'line' && state.linePoints) {
    let best = null, bestD = HANDLE_HIT_PX;
    state.linePoints.forEach((ln, i) => {
      ln.forEach(([x, y], ei) => {
        const d = Math.hypot(x * v.scale + v.tx - mx, y * v.scale + v.ty - my);
        if (d < bestD) { bestD = d; best = { mode: 'line-end', i, ei }; }
      });
    });
    return best;
  }

  return null;
}

function clampPointToImage(p) {
  if (!state.img) return p;
  return [
    Math.min(state.img.width - 1, Math.max(0, p[0])),
    Math.min(state.img.height - 1, Math.max(0, p[1])),
  ];
}

function bindCanvas() {
  const cv = $('c-source');

  cv.addEventListener('mousedown', (e) => {
    if (!state.img) return;
    const hit = hitTest(e.clientX, e.clientY);
    if (hit) {
      state.selected = (hit.mode === 'corner') ? hit.i : `l${hit.i}e${hit.ei}`;
      state.drag = { mode: hit.mode, i: hit.i, ei: hit.ei };
    } else {
      state.selected = -1;
      state.drag = { mode: 'pan', ox: e.clientX, oy: e.clientY,
        tx0: state.view.tx, ty0: state.view.ty };
    }
    drawSource();
  });

  window.addEventListener('mousemove', (e) => {
    if (state.img) {
      const [ix, iy] = toImage(e.clientX, e.clientY);
      const inside = ix >= 0 && iy >= 0 && ix < state.img.width && iy < state.img.height;
      $('cursor-pos').textContent = inside
        ? `(${ix.toFixed(1)}, ${iy.toFixed(1)})` : '—';
    }
    if (state.drag) {
      if (state.drag.mode === 'corner') {
        state.quad[state.drag.i] = clampPointToImage(toImage(e.clientX, e.clientY));
        drawSource();
        schedulePreview();
      } else if (state.drag.mode === 'line-end') {
        state.linePoints[state.drag.i][state.drag.ei] =
          clampPointToImage(toImage(e.clientX, e.clientY));
        const next = quadFromLines(state.linePoints);
        if (next) state.quad = next;
        drawSource();
        schedulePreview();
      } else {
        state.view.tx = state.drag.tx0 + (e.clientX - state.drag.ox);
        state.view.ty = state.drag.ty0 + (e.clientY - state.drag.oy);
        drawSource();
      }
    }
  });

  window.addEventListener('mouseup', () => {
    if (state.drag && (state.drag.mode === 'corner' || state.drag.mode === 'line-end')) {
      schedulePreview(true);
    }
    state.drag = null;
    // 这里**不能**清放大窗：松手后点还是选中的，放大窗必须继续锁在它上面。
  });

  cv.addEventListener('wheel', (e) => {
    if (!state.img) return;
    e.preventDefault();
    const rect = cv.getBoundingClientRect();
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    zoomTo(state.view.scale * factor, e.clientX - rect.left, e.clientY - rect.top);
    drawSource();
  }, { passive: false });

  window.addEventListener('keydown', (e) => {
    // 画廊打开时方向键归它用，别同时去挪角点
    if (!$('gallery').hidden) return;
    if (e.target.tagName === 'INPUT') return;
    if (state.selected === -1 || !state.img) return;
    const step = e.shiftKey ? 10 : 1;
    const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0],
      ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
    if (!d) return;
    e.preventDefault();

    if (state.editMode === 'corner' && typeof state.selected === 'number') {
      state.quad[state.selected] = clampPointToImage([
        state.quad[state.selected][0] + d[0],
        state.quad[state.selected][1] + d[1],
      ]);
      drawSource();
      schedulePreview();
    } else if (state.editMode === 'line' && typeof state.selected === 'string') {
      const rest = state.selected.slice(1);
      const i = parseInt(rest.split('e')[0], 10);
      const ei = parseInt(rest.split('e')[1], 10);
      state.linePoints[i][ei] = clampPointToImage([
        state.linePoints[i][ei][0] + d[0],
        state.linePoints[i][ei][1] + d[1],
      ]);
      const next = quadFromLines(state.linePoints);
      if (next) state.quad = next;
      drawSource();
      schedulePreview();
    }
  });

  new ResizeObserver(() => { drawSource(); }).observe($('canvas-wrap'));
}

/** 当前锁定点的原图坐标；未选中 / 数据缺失都返回 null。

    唯一来源是 state.selected：corner 模式是角点下标（-1 = 未选中），
    line 模式是 `l{i}e{ei}` 字符串。没有第二份"放大镜自己的目标"状态。
 */
function selectedImagePoint() {
  const sel = state.selected;
  if (state.editMode === 'corner') {
    if (typeof sel !== 'number' || sel < 0 || !state.quad) return null;
    return state.quad[sel] || null;
  }
  if (typeof sel !== 'string' || !state.linePoints) return null;
  const m = /^l(\d+)e(\d+)$/.exec(sel);
  if (!m) return null;
  const ln = state.linePoints[Number(m[1])];
  return ln ? (ln[Number(m[2])] || null) : null;
}

/** 放大窗：固定钉在源图区域右上角的 160×160 小窗，5x，中心锁定当前选中的点。

    位置不随鼠标动，只有内容会变：换选中点 → 换中心；拖选中点 → 实时重采样。
    未选中任何点时整个窗隐藏。窗内叠加主画布同一套几何绘制，中心画十字准星。
 */
function updateLoupe() {
  const loupe = $('c-loupe');
  if (!loupe) return;
  const target = state.img ? selectedImagePoint() : null;
  if (!target) { clearLoupe(); return; }

  const dpr = window.devicePixelRatio || 1;
  const px = Math.round(LOUPE_SIZE * dpr);
  if (loupe.width !== px || loupe.height !== px) {
    loupe.width = px;
    loupe.height = px;
  }
  loupe.hidden = false;

  const ctx = loupe.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, LOUPE_SIZE, LOUPE_SIZE);
  ctx.fillStyle = '#202020';
  ctx.fillRect(0, 0, LOUPE_SIZE, LOUPE_SIZE);

  const [ix, iy] = target;
  const c = LOUPE_SIZE / 2;
  const halfSrc = c / LOUPE_SCALE;

  // 源矩形可能越出图像边界；drawImage 会按同比例裁剪目标矩形，
  // 所以锁定点始终落在窗心，靠边时只是有一侧留空。
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(state.img, ix - halfSrc, iy - halfSrc, 2 * halfSrc, 2 * halfSrc,
    0, 0, LOUPE_SIZE, LOUPE_SIZE);

  // 窗内几何：复用主画布的绘制函数，把 toScreen 换成"锁定点落在窗心、其余按 5x 放大"。
  // 半透明画，否则窗心那个实心手柄正好盖住你要对的像素。
  const toScreenZ = (p) => [
    (p[0] - ix) * LOUPE_SCALE + c,
    (p[1] - iy) * LOUPE_SCALE + c,
  ];
  ctx.save();
  ctx.globalAlpha = 0.55;
  if (state.editMode === 'corner' && state.quad) {
    drawCornerMode(ctx, state.quad.map(toScreenZ));
  } else if (state.editMode === 'line' && state.linePoints) {
    const lines = state.linePoints.map((ln) => ln.map(toScreenZ));
    const corners = quadFromLines(state.linePoints);
    drawLineMode(ctx, lines, corners ? corners.map(toScreenZ) : null);
  }
  ctx.restore();

  // 十字准星：中间留空，别把要对的那个像素盖住
  ctx.strokeStyle = 'rgba(255,80,80,0.95)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(c - 22, c); ctx.lineTo(c - 5, c);
  ctx.moveTo(c + 5, c); ctx.lineTo(c + 22, c);
  ctx.moveTo(c, c - 22); ctx.lineTo(c, c - 5);
  ctx.moveTo(c, c + 5); ctx.lineTo(c, c + 22);
  ctx.stroke();
}

/** 隐藏并清空放大窗。唯一调用点在 updateLoupe 的"未选中"分支。 */
function clearLoupe() {
  const loupe = $('c-loupe');
  if (!loupe) return;
  loupe.hidden = true;
  const ctx = loupe.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, loupe.width, loupe.height);
}

// ---------------------------------------------------------------- 预览

function previewParams() {
  return {
    quad: state.quad,
    phys_w: parseFloat($('in-phys-w').value),
    phys_h: parseFloat($('in-phys-h').value),
    anchor_x: parseFloat($('in-ax').value),
    anchor_y: parseFloat($('in-ay').value),
    heading: parseFloat($('in-hd').value),
    scale: $('in-sc-auto').checked ? null : parseFloat($('in-sc').value),
  };
}

/** 读当前阈值输入框，空或非法返回 null。 */
function currentThreshold() {
  const v = parseFloat($('in-reproj').value);
  return (isFinite(v) && v > 0) ? v : null;
}

/** 打表选项。宽高都填了才降采样，否则保持原尺寸。 */
function tableOptions() {
  const w = parseInt($('in-table-w').value, 10);
  const h = parseInt($('in-table-h').value, 10);
  return {
    table_format: $('in-table-format').value,
    table_fixed_point: parseInt($('in-table-fp').value, 10),
    table_size: (w > 0 && h > 0) ? [w, h] : null,
  };
}

function schedulePreview(immediate) {
  clearTimeout(state.previewTimer);
  state.previewTimer = setTimeout(runPreview, immediate ? 0 : 150);
}

async function runPreview() {
  if (!state.img || !state.quad || state.previewing) return;
  state.previewing = true;
  const chip = $('preview-state');
  setChip(chip, '计算中…');
  try {
    const data = await api('/api/preview', previewParams());
    state.lastPreview = data;
    await drawPreview(data.birdview);
    const over = data.over_crop;
    setChip(chip, over ? '超出不裁切上限' : '正常', over ? 'warn' : 'ok');
    kv($('scale-info'), [
      ['当前 scale', `${data.scale.toFixed(3)} px/cm`],
      ['不裁切上限', `${data.max_scale.toFixed(3)} px/cm`],
      ['覆盖范围', `${($('in-phys-w').value / 1)} × ${($('in-phys-h').value / 1)} cm`],
    ], '');
    $('scale-info').lastChild.lastChild.className = over ? 'bad' : 'good';
    if ($('in-sc-auto').checked && !state.draggingScale) {
      $('in-sc').value = data.scale.toFixed(2);
      $('out-sc').textContent = data.scale.toFixed(2);
    }
  } catch (err) {
    // 四点退化是拖拽时最常见的即时反馈，其余失败多半是"还没选原图"之类，
    // 两者要能一眼区分，所以按错误内容给不同的标签。
    const geometry = /四边形|退化|凸|自交/.test(err.message);
    setChip(chip, geometry ? '四点无效' : '预览失败', 'bad');
    chip.title = err.message;
    $('scale-info').innerHTML = '';
  } finally {
    state.previewing = false;
  }
}

function drawPreview(dataUrl) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const cv = $('c-preview');
      const rect = $('preview-wrap').getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.round(rect.width));
      const h = Math.max(1, Math.round(rect.height));
      cv.width = w * dpr;
      cv.height = h * dpr;
      const ctx = cv.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = '#202020';
      ctx.fillRect(0, 0, w, h);
      const s = Math.min(w / img.width, h / img.height);
      ctx.imageSmoothingEnabled = s < 2.5;
      ctx.drawImage(img, (w - img.width * s) / 2, (h - img.height * s) / 2,
        img.width * s, img.height * s);
      resolve();
    };
    img.onerror = () => resolve();
    img.src = dataUrl;
  });
}

function bindControls() {
  ['in-phys-w', 'in-phys-h'].forEach((id) => $(id).addEventListener('input', () => schedulePreview()));

  // 改剔除阈值立刻重画柱状图与数量，让用户先看到"会删多少张"再决定跑不跑标定
  $('in-reproj').addEventListener('input', () => {
    if (!state.fit) return;
    drawErrorChart(state.fit, currentThreshold());
  });

  [['in-ax', 'out-ax', 3], ['in-ay', 'out-ay', 3], ['in-hd', 'out-hd', 1]]
    .forEach(([sid, oid, digits]) => {
      $(sid).addEventListener('input', () => {
        $(oid).textContent = parseFloat($(sid).value).toFixed(digits);
        schedulePreview();
      });
    });

  const sc = $('in-sc');
  sc.addEventListener('pointerdown', () => { state.draggingScale = true; });
  window.addEventListener('pointerup', () => { state.draggingScale = false; });
  sc.addEventListener('input', () => {
    $('out-sc').textContent = parseFloat(sc.value).toFixed(2);
    if ($('in-sc-auto').checked) $('in-sc-auto').checked = false;
    schedulePreview();
  });
  $('in-sc-auto').addEventListener('change', () => schedulePreview());

  $('btn-zoom-fit').onclick = () => { fitView(); drawSource(); };
  $('btn-zoom-100').onclick = () => {
    if (!state.img) return;
    const { w, h } = canvasMetrics();
    state.view.scale = 1;
    state.view.tx = (w - state.img.width) / 2;
    state.view.ty = (h - state.img.height) / 2;
    drawSource();
  };

  // 复位：回到载入这张图时服务端给的起点
  $('btn-reset-quad').onclick = () => {
    if (!state.quadGuess) return;
    if (state.editMode === 'line') {
      state.linePoints = linesFromQuad(state.quadGuess);
    } else {
      state.quad = state.quadGuess.map((p) => p.slice());
    }
    state.selected = -1;
    drawSource();
    schedulePreview(true);
  };

  // 载入上次结果：恢复上次成功导出时的四点，换图重做时不用重新拖一遍
  $('btn-guess-quad').onclick = () => {
    if (!state.committedQuad) {
      log('还没有可载入的历史四点，先做一次导出。', 'err');
      return;
    }
    if (state.editMode === 'line') {
      state.linePoints = linesFromQuad(state.committedQuad);
    } else {
      state.quad = state.committedQuad.map((p) => p.slice());
    }
    state.selected = -1;
    drawSource();
    schedulePreview(true);
    log('已载入上次导出的四点。');
  };

  // 模式切换：四角点 ↔ 交点
  const setMode = (mode) => {
    if (state.editMode === mode) return;
    // 切换前先把当前的角点同步到 linePoints（反之亦然），两种模式共享同一组角点
    if (mode === 'line') {
      if (!state.linePoints) state.linePoints = linesFromQuad(state.quad);
    } else {
      if (state.linePoints) {
        const next = quadFromLines(state.linePoints);
        if (next) state.quad = next;
      }
    }
    state.editMode = mode;
    state.selected = -1;
    $('btn-mode-corner').classList.toggle('active', mode === 'corner');
    $('btn-mode-line').classList.toggle('active', mode === 'line');
    drawSource();
    schedulePreview();
  };
  $('btn-mode-corner').onclick = () => setMode('corner');
  $('btn-mode-line').onclick = () => setMode('line');

  // 文件夹选择：浏览器出于安全不给真实磁盘路径，只给文件内容 + 相对路径。
  // 所以这里把选中的文件整个留着，点「开始导入」时直接上传，
  // 而不是把文件夹名丢给服务端去磁盘上猜位置（改名/挪地方必然失败）。
  $('btn-browse').onclick = () => $('in-import-browse').click();
  $('in-import-browse').onchange = (e) => {
    const picked = Array.from(e.target.files || [])
      .filter(f => /\.(jpe?g|png|bmp)$/i.test(f.name));
    if (!picked.length) return;
    pendingImportFiles = picked;
    // webkitRelativePath 形如 "MyFolder/sub/file.jpg"，目录是第一段
    const dir = (picked[0].webkitRelativePath || picked[0].name).split('/')[0];
    $('in-import-dir').value = dir;
    log(`已选择 ${picked.length} 个文件，点「开始导入」会上传到 data/import/${dir}/`, 'ok');
  };
}

// ---------------------------------------------------------------- 成果图画廊

const gallery = { items: [], index: -1 };

/** 打开去畸变成果图画廊。 */
async function openGallery() {
  let data;
  try {
    data = await api('/api/preview_gallery');
  } catch (e) {
    log('读取成果图失败: ' + e.message, 'err');
    return;
  }
  gallery.items = data.items || [];
  gallery.index = -1;

  const grid = $('gallery-grid');
  grid.innerHTML = '';
  if (!gallery.items.length) {
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = 'calib_preview/ 还是空的。跑一次「运行标定」就会在这里生成每张标定图的去畸变结果。';
    grid.appendChild(p);
  } else {
    gallery.items.forEach((it, i) => {
      const el = document.createElement('div');
      el.className = 'shot';
      el.title = it.name;
      const img = document.createElement('img');
      img.loading = 'lazy';
      // 缩略图优先用去畸变图；只剩原图时也得让这一组可点开，
      // 但要在标签上说清楚，别让人把原图当成去畸变成果验收了。
      img.src = thumbUrl(it.undistorted_url || it.raw_url, 360);
      const span = document.createElement('span');
      span.textContent = it.undistorted_url ? it.name : it.name + '（缺去畸变）';
      el.append(img, span);
      el.onclick = () => showGalleryItem(i);
      grid.appendChild(el);
    });
  }

  $('gallery-count').textContent = String(gallery.items.length);
  $('gallery-viewer').hidden = true;
  $('gallery').hidden = false;
  log(`打开成果图画廊：${gallery.items.length} 组对照，`
    + `其中 ${data.paired} 组两侧齐全`
    + `（缺原图 ${data.missing_raw}，缺去畸变 ${data.missing_undistorted}）。`);
}

/** 给服务端下发的取图地址补上最大边限制。 */
function thumbUrl(url, maxEdge) {
  return url + '&max=' + maxEdge;
}

function closeGallery() {
  $('gallery').hidden = true;
  $('gallery-viewer').hidden = true;
}

/** 在放大视图里显示第 i 张，并同步网格高亮。 */
function showGalleryItem(i) {
  if (i < 0 || i >= gallery.items.length) return;
  gallery.index = i;
  const it = gallery.items[i];

  setPane('viewer-source', it.raw_url);
  setPane('viewer-preview', it.undistorted_url);
  $('viewer-name').textContent = `${it.name}  ·  ${i + 1} / ${gallery.items.length}`;
  $('gallery-viewer').hidden = false;

  document.querySelectorAll('#gallery-grid .shot').forEach((el, k) => {
    el.classList.toggle('active', k === i);
  });
  const active = document.querySelector('#gallery-grid .shot.active');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

/** 装一侧的图；地址为空就清掉 src 并露出空态占位。

    清 src 这一步是硬要求：留着上一张的 src 的话，缺失的那一侧会继续显示
    上一组的图，看上去就是"A 的原图配 B 的去畸变图"——比什么都不显示更糟。
 */
function setPane(imgId, url) {
  const img = $(imgId);
  const miss = $(imgId + '-missing');
  if (url) {
    img.src = thumbUrl(url, 1600);
    img.hidden = false;
    miss.hidden = true;
  } else {
    img.removeAttribute('src');
    img.hidden = true;
    miss.hidden = false;
  }
}

function galleryStep(delta) {
  if (gallery.index < 0) return;
  const n = gallery.items.length;
  showGalleryItem((gallery.index + delta + n) % n);
}

/** 绑定画廊的按钮与快捷键。 */
function bindGallery() {
  $('btn-gallery').onclick = () => withBusy($('btn-gallery'), openGallery);
  $('gallery-close').onclick = closeGallery;
  $('viewer-prev').onclick = () => galleryStep(-1);
  $('viewer-next').onclick = () => galleryStep(1);
  // 点遮罩空白处关闭（点面板内部不关）
  $('gallery').addEventListener('mousedown', (e) => {
    if (e.target === $('gallery')) closeGallery();
  });

  window.addEventListener('keydown', (e) => {
    if ($('gallery').hidden) return;
    if (e.key === 'Escape') { closeGallery(); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { galleryStep(-1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { galleryStep(1); e.preventDefault(); }
  });
}

/** 跑一次相机标定并把结果画到卡片上。

    threshold 用 undefined 表示"沿用输入框里的值"，传 '' 或 null 表示强制不剔除。
    导入素材后会自动调一次（不剔除），目的只是先拿到全体误差分布，
    让用户能看着柱状图挑阈值，省掉"先盲标一次 → 调阈值 → 再标一次"的第一遍。
 */
async function runCalibration({ force = false, threshold } = {}) {
  const payload = {
    force,
    max_reproj_err: threshold === undefined ? $('in-reproj').value : (threshold ?? ''),
    board: boardPayload(),
  };
  const data = await api('/api/calibrate', payload);
  log(data.log);
  if (data.fit) {
    state.fit = data.fit;
    drawErrorChart(data.fit, currentThreshold());
  } else {
    $('calib-count').textContent = '（仅复用了已存的 calib.json，未重跑标定）';
    $('calib-count').className = 'hint mono';
  }
  // 必须在 state.fit 设好之后再刷新：refreshStatus 末尾会用 !state.fit 决定
  // 要不要把图表切成空态，顺序反了会把刚画好的柱状图清掉。
  // 这一步同时更新状态徽标、内参摘要和"成果图（N 张）"的徽标——
  // 漏掉它就会出现"图有了但徽标还是暂无"。
  await refreshStatus();
  return data;
}

/** 刷新"已有几个备份"的提示行。 */
async function refreshBackups() {
  const el = $('backup-info');
  try {
    const d = await api('/api/backup_list');
    if (!d.count) {
      el.textContent = '尚无备份。';
      el.className = 'hint mono';
      return;
    }
    const last = d.items[0];
    el.textContent = `已有 ${d.count} 个备份，最近：${last.name}`
      + (last.created ? `（${last.created}）` : '');
    el.className = 'hint mono warn';
  } catch (e) {
    el.textContent = '';
  }
}

// ---------------------------------------------------------------- 动作

// 浏览器选中、等待上传的素材文件（见 bindActions 里的文件夹选择）
let pendingImportFiles = [];

function updateBoardHint() {
  const x = parseInt($('in-board-x').value, 10);
  const y = parseInt($('in-board-y').value, 10);
  const cx = x - 1;
  const cy = y - 1;
  // 与后端 CheckerboardSpec 用同一判据：内角点短边 ≥3、长边 ≥4。
  // 刻意不假设"横向一定多于纵向"——4x5 与 5x4 只是转了 90°，应当等价。
  const ok = Number.isFinite(cx) && Number.isFinite(cy)
    && Math.min(cx, cy) >= 3 && Math.max(cx, cy) >= 4;
  $('board-corners').textContent = ok ? `${cx} × ${cy}` : '内角点至少 3×4（方格 4×5）';
  $('board-corners').style.color = ok ? '' : 'var(--danger)';
}

// 规格是"一张图算棋盘照还是地面照"的依据，所以导入/标定请求都带上它，
// 免得用户改了网页却没生效（服务端会以请求里的值为准）。
function boardPayload() {
  return {
    squares_x: parseInt($('in-board-x').value, 10),
    squares_y: parseInt($('in-board-y').value, 10),
    square_size_mm: parseFloat($('in-board-mm').value),
  };
}

function bindActions() {
  $('btn-refresh').onclick = () => withBusy($('btn-refresh'), async () => {
    try { await refreshStatus(); } catch (e) { log('扫描失败: ' + e.message, 'err'); }
  });

  // 退出：走服务端的 /api/shutdown，用户就不必去命令行按 Ctrl+C
  //（Windows 下 cmd.exe 还会追问一句 "Terminate batch job (Y/N)?"）
  $('btn-quit').onclick = async () => {
    if (!confirm('确定退出本地控制台吗？未保存的交互标定不会丢失，但页面需要重新启动程序才能再用。')) return;
    try {
      await api('/api/shutdown', {});
      document.body.innerHTML =
        '<div class="quit-screen"><h2>控制台已退出</h2>' +
        '<p>可以关掉这个标签页和黑色命令行窗口了。</p></div>';
    } catch (e) {
      log('退出失败: ' + e.message, 'err');
    }
  };

  $('btn-board-apply').onclick = () => withBusy($('btn-board-apply'), async () => {
    try {
      const b = await api('/api/board', boardPayload());
      $('board-active').textContent = '当前生效：' + b.label;
      log(`标定板规格已设为 ${b.label}（OpenCV 内角点 ${b.corners_x}×${b.corners_y}）`
        + '，已写入 project.json', 'ok');
      // 规格参与"棋盘照/地面照"的判定：改了规格，已导入的素材就是按旧标准分的
      renderMaterialWarning(b.material_stale);
      renderTransactionWarning(b.material_transaction);
      if (b.material_stale) log('⚠ ' + b.material_stale, 'err');
    } catch (e) { log('规格设置失败: ' + e.message, 'err'); }
  });

  $('btn-board-reset').onclick = () => withBusy($('btn-board-reset'), async () => {
    try {
      const b = await api('/api/board',
        { squares_x: 12, squares_y: 9, square_size_mm: 20 });
      $('in-board-x').value = b.squares_x;
      $('in-board-y').value = b.squares_y;
      $('in-board-mm').value = b.square_size_mm;
      updateBoardHint();
      $('board-active').textContent = '当前生效：' + b.label;
      renderMaterialWarning(b.material_stale);
      renderTransactionWarning(b.material_transaction);
      log('已恢复为仓库自带棋盘 12×9 / 20 mm', 'ok');
    } catch (e) { log('恢复失败: ' + e.message, 'err'); }
  });

  ['in-board-x', 'in-board-y', 'in-board-mm'].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener('input', updateBoardHint);
  });

  $('btn-import').onclick = () => withBusy($('btn-import'), async () => {
    try {
      const mode = document.querySelector('input[name="import-mode"]:checked').value;
      // 用"选择文件夹"选的：直接把文件传上去，不走路径查找
      let data;
      if (pendingImportFiles && pendingImportFiles.length) {
        const fd = new FormData();
        fd.append('name', $('in-import-dir').value.trim());
        // 规格与模式都必须跟着走：漏掉 mode 会让"覆盖整个素材库"
        // 只对填路径导入生效，走上传时静默变回增量添加。
        fd.append('squares_x', $('in-board-x').value);
        fd.append('squares_y', $('in-board-y').value);
        fd.append('square_size_mm', $('in-board-mm').value);
        fd.append('mode', mode);
        pendingImportFiles.forEach(f => {
          fd.append('files', f, f.webkitRelativePath || f.name);
        });
        data = await upload('/api/upload_import', fd);
        pendingImportFiles = [];
      } else {
        data = await api('/api/import', {
          dir: $('in-import-dir').value,
          mode,
          move: $('in-import-move').checked,
          board: boardPayload(),
        });
      }
      log(data.log);
      const s = data.summary;
      if (s) {
        const bits = [`棋盘 ${s.imported_calib} 张`];
        if (s.imported_partial) bits.push(`棋盘不全 ${s.imported_partial} 张`);
        bits.push(`地面 ${s.imported_ipm} 张`);
        if (s.skipped_duplicates) bits.push(`重复跳过 ${s.skipped_duplicates} 张`);
        if (s.cleared) bits.push('已整体替换原素材库');
        log('导入摘要: ' + bits.join('，'),
          (s.imported_calib || s.imported_ipm || s.imported_partial) ? 'ok' : 'err');
      }
      log('导入完成。', 'ok');
      const st = await refreshStatus();

      // 导入完自动先标一次（不剔除），把全体误差分布先算出来。
      // 否则用户要"盲标一次 → 看误差挑阈值 → 再标一次"，白跑一遍。
      const nCalib = (st.photo_counts && st.photo_counts.calib) || 0;
      if (nCalib >= 3) {
        log(`检测到 ${nCalib} 张标定照片，自动先跑一次标定以生成误差分布…`);
        await runCalibration({ force: true, threshold: '' });
        log('自动标定完成。现在可以直接看柱状图挑阈值，'
          + '改完点「运行标定」即可生效。', 'ok');
      } else if (nCalib > 0) {
        log(`标定照片只有 ${nCalib} 张（至少需要 3 张），跳过自动标定。`, 'err');
      } else {
        log('没有导入到棋盘照片，跳过自动标定。');
      }
    } catch (e) { log('导入失败: ' + e.message, 'err'); }
  });

  $('btn-calib').onclick = () => withBusy($('btn-calib'), async () => {
    try {
      log('开始相机标定…');
      await runCalibration({ force: $('in-force').checked });
      log('标定完成。', 'ok');
    } catch (e) { log('标定失败: ' + e.message, 'err'); }
  });

  $('btn-commit').onclick = () => withBusy($('btn-commit'), async () => {
    if (!state.quad) { log('请先选择一张原图并调整四点。', 'err'); return; }
    if ($('in-sc-auto').checked === false && state.lastPreview && state.lastPreview.over_crop) {
      log('警告: 当前 scale 超出不裁切上限，导出的表会裁掉部分有效视野。', 'err');
    }
    try {
      const opts = tableOptions();
      log(`导出中：表格式 ${opts.table_format}`
        + (opts.table_size ? `，网格 ${opts.table_size[0]}x${opts.table_size[1]}` : '，原尺寸')
        + (opts.table_format === 'txt' ? '' : `，Q${opts.table_fixed_point}`));
      const data = await api('/api/commit', { ...previewParams(), ...opts });
      state.committedQuad = state.quad.map((p) => p.slice());
      log(data.log);
      log('导出完成，产物: ' + data.files.map((f) => f.name).join(', '), 'ok');
      await refreshStatus();
    } catch (e) { log('导出失败: ' + e.message, 'err'); }
  });

  $('btn-batch').onclick = () => withBusy($('btn-batch'), async () => {
    try {
      const data = await api('/api/batch', {});
      log(data.log);
      log('批量测试完成。', 'ok');
      await refreshStatus();   // test_output 的张数要跟着更新
    } catch (e) { log('批量测试失败: ' + e.message, 'err'); }
  });

  $('btn-backup-clear').onclick = () => {
    const name = $('in-backup-name').value.trim();
    const shown = name || '（留空 → 用打包时间命名）';
    const ok = window.confirm(
      '确认「备份并清空」？\n\n'
      + '会把下列目录打包成一个 zip 放进 backups/：\n'
      + '  calib_input、ipm_input、test_input\n'
      + '  calib_data、calib_preview、ipm_output\n'
      + '  matrix、lookup_table、test_output\n\n'
      + '打包成功后会清空这些目录，等待新素材导入。\n'
      + '（matrix/legacy_matlab_reference.json 和 calib_data/calibrationSession.mat\n'
      + '  属于参考资料，会保留，备份里也有一份）\n'
      + '打包或校验失败则直接中止，不动任何文件。\n\n'
      + '备份名：' + shown);
    if (!ok) return;

    withBusy($('btn-backup-clear'), async () => {
      try {
        const mb = (n) => (n / 1024 / 1024).toFixed(1) + ' MB';
        log('正在打包备份…');
        const data = await api('/api/backup_clear', { name });
        log(`备份已写入: ${data.path}`);
        data.folders.forEach((f) => log(`    清空 ${f.folder}: ${f.files} 个文件`));
        log(`  压缩包 ${mb(data.archive_bytes)}（原始 ${mb(data.source_bytes)}）`);
        if (data.kept_files) {
          log(`  保留 ${data.kept_files} 个参考资料文件（MATLAB 参照矩阵、标定会话原始文件）`);
        }
        log(`已清空 ${data.cleared_files} 个文件，各目录已就位，等待新素材导入。`, 'ok');
        $('in-backup-name').value = '';
        state.fit = null;
        state.committedQuad = null;
        drawErrorChart(null, null);
        await refreshStatus();
        await refreshBackups();
      } catch (e) {
        log('备份并清空失败（原目录未改动）: ' + e.message, 'err');
      }
    });
  };

  $('btn-clear-all').onclick = () => {
    const ok = window.confirm(
      '⚠️ 仅清空，不备份 —— 此操作不可撤销\n\n'
      + '会直接删除下列目录里的全部文件：\n'
      + '  calib_input、ipm_input、test_input\n'
      + '  calib_data、calib_preview、ipm_output\n'
      + '  matrix、lookup_table、test_output\n\n'
      + '标定结果、查找表、去畸变成果图都会没有，且没有备份可以还原。\n'
      + '（matrix/legacy_matlab_reference.json 和 calib_data/calibrationSession.mat\n'
      + '  属于参考资料，会保留）\n\n'
      + '如果只是想换一批素材，建议改用「备份并清空」。\n\n'
      + '确定要清空吗？');
    if (!ok) return;

    withBusy($('btn-clear-all'), async () => {
      try {
        // confirm:true 是服务端的硬性要求，不带就直接 400 拒掉
        const data = await api('/api/clear_all', { confirm: true });
        log(`已清空 ${data.cleared_files} 个文件（未备份，无法还原）。`, 'err');
        if (data.kept_files) {
          log(`  保留 ${data.kept_files} 个参考资料文件。`);
        }
        state.fit = null;
        state.committedQuad = null;
        drawErrorChart(null, null);
        await refreshStatus();
      } catch (e) {
        log('清空失败（原目录未改动）: ' + e.message, 'err');
      }
    });
  };

  $('btn-clear-log').onclick = clearLog;
}

// ---------------------------------------------------------------- 启动

(async function boot() {
  bindCanvas();
  bindControls();
  bindActions();
  bindGallery();
  refreshBackups();
  try {
    const st = await refreshStatus();
    const saved = st.ipm_state || null;
    const savedName = saved && saved.src_image
      ? saved.src_image.split(/[\\/]/).pop() : null;

    // 上次用的原图若还在候选里就优先选它，否则退回第一张
    const pick = (savedName && st.ipm_candidates.includes(savedName))
      ? savedName
      : (st.ipm_candidates[0] || null);

    if (saved && Array.isArray(saved.src_quad_tl_tr_bl_br)) {
      state.committedQuad = saved.src_quad_tl_tr_bl_br.map((p) => p.slice());
    }
    if (!pick) {
      log('ipm_input/ 里还没有图片，先在第 1 步导入素材。', 'err');
      return;
    }

    await loadSource(pick);

    // 只有"当前选的正是上次那张图"时，历史参数才适用
    if (saved && savedName === pick) {
      $('in-phys-w').value = saved.phys_w_cm;
      $('in-phys-h').value = saved.phys_h_cm;
      $('in-ax').value = saved.anchor_x;
      $('out-ax').textContent = Number(saved.anchor_x).toFixed(3);
      $('in-ay').value = saved.anchor_y;
      $('out-ay').textContent = Number(saved.anchor_y).toFixed(3);
      $('in-hd').value = saved.heading_deg;
      $('out-hd').textContent = Number(saved.heading_deg).toFixed(1);
      if (Array.isArray(saved.src_quad_tl_tr_bl_br)) {
        state.quad = saved.src_quad_tl_tr_bl_br.map((p) => p.slice());
        state.quadGuess = state.quad.map((p) => p.slice());
        drawSource();
      }
      schedulePreview(true);
      log('已恢复上次的逆透视标定参数与四点。');
    }
  } catch (e) {
    log('初始化失败: ' + e.message, 'err');
  }
})();

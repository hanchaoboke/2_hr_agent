/* ==========================================================================
   飞羽科技 · HR 智能助理 — 前端逻辑
   职责：SSE 流式解析、工具台账渲染、markdown 渲染、表单验证、健康检查
   ========================================================================== */

"use strict";

/* ---------- 配置 ---------- */
// 后端 API 地址，可通过 localStorage 的 hr_agent_api_base 覆盖
const API_BASE =
  localStorage.getItem("hr_agent_api_base") || "http://192.168.10.229:8000";

/* ---------- 工具元数据（中文标签 + 图标） ---------- */
const TOOL_META = {
  get_employee_profile: { label: "员工档案", icon: "👤" },
  get_leave_balance: { label: "假期余额", icon: "🏖️" },
  generate_employment_certificate: { label: "证明开具", icon: "📄" },
  search_hr_policy: { label: "政策检索", icon: "🧾" },
};

/* ---------- 状态 ---------- */
const state = {
  threadId: randomId(),
  streaming: false,
  pendingReview: false,
};

/* ---------- DOM 引用 ---------- */
const $ = (sel) => document.querySelector(sel);
const messagesEl = $("#messages");
const formEl = $("#form");
const inputEl = $("#input");
const sendBtn = $("#sendBtn");
const uidInput = $("#uid");
const threadIdEl = $("#threadId");
const healthEl = $("#health");
const sidebarEl = $("#sidebar");
const scrimEl = $("#scrim");
const menuBtn = $("#menuBtn");
const newThreadBtn = $("#newThread");
const clearBtn = $("#clearBtn");
const samplesEl = $("#samples");
const toastEl = $("#toast");

/* ==========================================================================
   小工具
   ========================================================================== */
function el(tag, cls) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  return node;
}

function randomId() {
  return Math.random().toString(16).slice(2, 10);
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function scrollToBottom() {
  const sc = $("#scroll");
  sc.scrollTop = sc.scrollHeight;
}

/* ---------- 全局提示 ---------- */
let toastTimer = null;
function toast(msg, type) {
  toastEl.textContent = msg;
  toastEl.className = "toast toast--show" + (type === "err" ? " toast--err" : "");
  toastEl.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastEl.className = "toast";
    toastEl.hidden = true;
  }, 2600);
}

/* ==========================================================================
   Markdown 渲染（安全：先转义，再处理块级 / 行内）
   ========================================================================== */
function inlineMd(text) {
  // 先保护行内代码，避免被后续替换破坏
  const codes = [];
  text = text.replace(/`([^`]+)`/g, (m, c) => {
    codes.push("<code>" + escapeHtml(c) + "</code>");
    return "\x00" + (codes.length - 1) + "\x00";
  });

  text = escapeHtml(text);
  // 链接 [文字](url)
  text = text.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );
  // 加粗 / 斜体
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  // 还原行内代码
  text = text.replace(/\x00(\d+)\x00/g, (m, i) => codes[+i]);
  return text;
}

function renderMarkdown(md) {
  if (!md) return "";
  const lines = md.split("\n");
  let html = "";
  let inList = null;
  let inCode = false;
  let codeBuf = [];
  let para = [];

  const flushPara = () => {
    if (para.length) {
      html += "<p>" + inlineMd(para.join(" ")) + "</p>";
      para = [];
    }
  };
  const closeList = () => {
    if (inList) {
      html += "</" + inList + ">";
      inList = null;
    }
  };

  for (const raw of lines) {
    // 代码围栏
    if (raw.trim().startsWith("```")) {
      if (inCode) {
        html += "<pre><code>" + escapeHtml(codeBuf.join("\n")) + "</code></pre>";
        codeBuf = [];
        inCode = false;
      } else {
        flushPara();
        closeList();
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      codeBuf.push(raw);
      continue;
    }

    const t = raw.trim();
    if (!t) {
      flushPara();
      closeList();
      continue;
    }
    // 标题
    const h = t.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      flushPara();
      closeList();
      const lv = h[1].length;
      html += "<h" + lv + ">" + inlineMd(h[2]) + "</h" + lv + ">";
      continue;
    }
    // 分隔线
    if (/^(-{3,}|\*{3,})$/.test(t)) {
      flushPara();
      closeList();
      html += "<hr>";
      continue;
    }
    // 引用
    if (t.startsWith(">")) {
      flushPara();
      closeList();
      html += "<blockquote>" + inlineMd(t.replace(/^>\s?/, "")) + "</blockquote>";
      continue;
    }
    // 无序列表
    const ul = t.match(/^[-*+]\s+(.*)$/);
    if (ul) {
      flushPara();
      if (inList !== "ul") {
        closeList();
        html += "<ul>";
        inList = "ul";
      }
      html += "<li>" + inlineMd(ul[1]) + "</li>";
      continue;
    }
    // 有序列表
    const ol = t.match(/^\d+[.)]\s+(.*)$/);
    if (ol) {
      flushPara();
      if (inList !== "ol") {
        closeList();
        html += "<ol>";
        inList = "ol";
      }
      html += "<li>" + inlineMd(ol[1]) + "</li>";
      continue;
    }
    // 普通段落
    closeList();
    para.push(t);
  }

  flushPara();
  closeList();
  if (inCode && codeBuf.length) {
    html += "<pre><code>" + escapeHtml(codeBuf.join("\n")) + "</code></pre>";
  }
  return html;
}

/* ==========================================================================
   消息渲染
   ========================================================================== */
function appendUser(text) {
  const msg = el("div", "msg msg--user");
  const bubble = el("div", "msg__bubble");
  bubble.textContent = text;
  msg.appendChild(bubble);
  messagesEl.appendChild(msg);
  scrollToBottom();
}

/* 创建一个助理消息“回合”，流式过程中逐步填充 */
function createAssistantTurn() {
  const msg = el("div", "msg msg--assistant");
  const avatar = el("div", "msg__avatar");
  avatar.textContent = "羽";
  const body = el("div", "msg__body");
  msg.append(avatar, body);
  messagesEl.appendChild(msg);
  scrollToBottom();

  return {
    el: msg,
    body,
    ledger: null,
    pending: [], // 尚未完成的工具步骤
    answer: null,
    typing: null,
    foot: null,
    answered: false,
    lastContent: null,
    done: false,
  };
}

/* 打字指示器 */
function showTyping(turn) {
  turn.typing = el("div", "msg__bubble msg__bubble--typing");
  turn.typing.innerHTML = '<span class="td"></span><span class="td"></span><span class="td"></span>';
  turn.body.appendChild(turn.typing);
}

/* 台账（处理过程） */
function ensureLedger(turn) {
  if (turn.ledger) return turn.ledger;
  const ledger = el("div", "ledger");
  const head = el("div", "ledger__head");
  head.innerHTML = "处理过程 <span class=\"ledger__count\">0 步</span>";
  const steps = el("div", "ledger__steps");
  ledger.append(head, steps);
  turn.body.insertBefore(ledger, turn.typing || turn.body.firstChild);
  turn.ledger = {
    root: ledger,
    steps,
    countEl: head.querySelector(".ledger__count"),
    total: 0,
  };
  return turn.ledger;
}

function updateLedgerCount(l) {
  const done = l.steps.querySelectorAll(".step--done").length;
  l.countEl.textContent = done + "/" + l.total + " 步";
}

function addToolSteps(turn, tools) {
  const l = ensureLedger(turn);
  tools.forEach((tool) => {
    const meta = TOOL_META[tool] || { label: tool, icon: "🔧" };
    const step = el("div", "step step--active");
    const dot = el("span", "step__dot");
    const name = el("span", "step__name");
    name.textContent = meta.icon + " " + meta.label;
    const toolEl = el("span", "step__tool");
    toolEl.textContent = tool;
    const stateEl = el("span", "step__state");
    stateEl.textContent = "执行中";
    step.append(dot, name, toolEl, stateEl);
    l.steps.appendChild(step);
    l.total++;
    turn.pending.push({ tool, step, stateEl });
  });
  updateLedgerCount(l);
  scrollToBottom();
}

function completeTool(turn, tool, preview) {
  // 从后往前找最近一个“执行中”的同名步骤
  for (let i = turn.pending.length - 1; i >= 0; i--) {
    const s = turn.pending[i];
    if (s.tool === tool && s.step.classList.contains("step--active")) {
      s.step.classList.remove("step--active");
      s.step.classList.add("step--done");
      s.stateEl.textContent = "完成";
      if (preview) {
        const p = el("div", "step__preview");
        p.textContent = truncate(preview, 96);
        s.step.after(p);
      }
      break;
    }
  }
  if (turn.ledger) updateLedgerCount(turn.ledger);
  scrollToBottom();
}

function renderAnswer(turn, content) {
  // 清理打字指示器
  if (turn.typing) {
    turn.typing.remove();
    turn.typing = null;
  }
  if (!turn.answer) {
    turn.answer = el("div", "msg__bubble msg__bubble--ai");
    turn.body.appendChild(turn.answer);
  }
  // 内容未变化（后端偶发重复帧）时跳过，避免误判为审计重写
  const changed = content !== turn.lastContent;
  if (turn.answered && changed) {
    const note = el("div", "step__preview");
    note.textContent = "↻ 已根据合规审计意见重写答复";
    note.style.marginLeft = "0";
    turn.answer.before(note);
  }
  if (changed) {
    turn.answer.innerHTML = renderMarkdown(content);
    turn.lastContent = content;
  }
  turn.answered = true;

  // 免责声明脚注（仅一次）
  if (!turn.foot) {
    turn.foot = el("div", "msg__foot");
    turn.foot.textContent = "此内容由 AI 生成，最终解释权归 HR 部门所有。";
    turn.body.appendChild(turn.foot);
  }
  scrollToBottom();
}

function showTurnError(turn, message) {
  if (turn.typing) {
    turn.typing.remove();
    turn.typing = null;
  }
  const err = el("div", "msg__bubble msg__bubble--err");
  err.textContent = "⚠️ " + message;
  turn.body.appendChild(err);
  scrollToBottom();
}

function finalizeTurn(turn) {
  if (turn.done) return;
  turn.done = true;
  if (turn.typing) {
    turn.typing.remove();
    turn.typing = null;
  }
  // 收尾：仍处于“执行中”的步骤去掉脉冲
  turn.pending.forEach((s) => {
    if (s.step.classList.contains("step--active")) {
      s.step.classList.remove("step--active");
      s.stateEl.textContent = "—";
    }
  });
}

/* ==========================================================================
   SSE 流式调用
   ========================================================================== */
async function streamChat(payload, turn) {
  let resp;
  try {
    resp = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error("无法连接服务，请确认后端已启动（默认 http://127.0.0.1:8000）");
  }

  if (!resp.ok) {
    let detail = "";
    try {
      const j = await resp.json();
      detail = j.detail ? " " + JSON.stringify(j.detail) : "";
    } catch {}
    throw new Error(`请求失败（HTTP ${resp.status}）${detail}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of raw.split("\n")) {
        const t = line.trim();
        if (!t.startsWith("data:")) continue;
        let frame;
        try {
          frame = JSON.parse(t.slice(5).trim());
        } catch {
          continue;
        }
        handleFrame(frame, turn);
      }
    }
  }
}

function handleFrame(frame, turn) {
  switch (frame.type) {
    case "tool_call":
      addToolSteps(turn, frame.tools || []);
      break;
    case "tool_result":
      completeTool(turn, frame.tool, frame.preview);
      break;
    case "answer":
      if (frame.content) renderAnswer(turn, frame.content);
      break;
    case "error":
      // 先记录错误，流结束后再探测是否属于「等待人工审批」场景
      turn.flowError = frame.message || "未知错误";
      break;
    case "done":
      finalizeTurn(turn);
      break;
    default:
      break;
  }
}

/* ==========================================================================
   人工审批
   ========================================================================== */
async function probeAfterStream(turn, payload) {
  // SSE 流未产出答案：可能是触发了人工审批（interrupt），用 JSON 接口探测状态
  let data = null;
  try {
    const resp = await fetch(`${API_BASE}/api/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    data = await resp.json().catch(() => null);
    if (!resp.ok || !data) {
      throw new Error((data && data.detail) || `请求失败（HTTP ${resp.status}）`);
    }
  } catch (e) {
    showTurnError(turn, (data && data.detail) || turn.flowError || e.message || "服务异常");
    finalizeTurn(turn);
    return;
  }

  if (data.code === 202) {
    // 图挂起，进入等待人工审批状态
    renderReviewCard(turn, data.interrupt_msg || "Agent 正在尝试执行敏感操作，等待人工授权。");
  } else if (data.code === 200 && data.answer) {
    finalizeTurn(turn);
    renderAnswer(turn, data.answer);
  } else {
    showTurnError(turn, turn.flowError || "服务未返回有效结果");
    finalizeTurn(turn);
  }
}

/* 渲染人工审批卡片（内联样式贴合现有设计令牌，不改动样式文件） */
function renderReviewCard(turn, message) {
  finalizeTurn(turn);
  state.pendingReview = true;

  const card = el("div");
  card.style.cssText =
    "border:1px solid var(--line);border-radius:12px;padding:14px 16px;background:var(--surface);";

  const title = el("p");
  title.style.cssText =
    "margin:0 0 6px;font-weight:700;font-size:14px;color:var(--seal);display:flex;align-items:center;gap:6px;";
  title.textContent = "🔒 需要人工审批";

  const desc = el("p");
  desc.style.cssText =
    "margin:0 0 14px;font-size:13.5px;line-height:1.7;color:var(--ink-2);white-space:pre-wrap;word-break:break-word;";
  desc.textContent = message;

  const actions = el("div");
  actions.style.cssText = "display:flex;gap:10px;";

  const approveBtn = el("button");
  approveBtn.type = "button";
  approveBtn.textContent = "✓ 批准";
  approveBtn.style.cssText =
    "flex:1;padding:9px 0;border-radius:9px;background:var(--pine);color:#fff;font-size:13.5px;font-weight:600;";
  approveBtn.addEventListener("mouseenter", () => {
    if (!approveBtn.disabled) approveBtn.style.background = "var(--pine-deep)";
  });
  approveBtn.addEventListener("mouseleave", () => {
    if (!approveBtn.disabled) approveBtn.style.background = "var(--pine)";
  });

  const rejectBtn = el("button");
  rejectBtn.type = "button";
  rejectBtn.textContent = "✕ 拒绝";
  rejectBtn.style.cssText =
    "flex:1;padding:9px 0;border-radius:9px;background:var(--seal-soft);color:var(--seal);font-size:13.5px;font-weight:600;border:1px solid var(--seal);";

  const setBusy = (busy) => {
    [approveBtn, rejectBtn].forEach((btn) => {
      btn.disabled = busy;
      btn.style.opacity = busy ? "0.6" : "1";
      btn.style.cursor = busy ? "not-allowed" : "pointer";
    });
  };

  actions.append(approveBtn, rejectBtn);
  card.append(title, desc, actions);
  turn.body.appendChild(card);
  scrollToBottom();

  approveBtn.addEventListener("click", () => submitReview(turn, card, desc, setBusy, "approve"));
  rejectBtn.addEventListener("click", () => submitReview(turn, card, desc, setBusy, "reject"));
}

/* 提交人工审批（批准 / 拒绝），调用 /api/v1/resume */
async function submitReview(turn, card, desc, setBusy, action) {
  setBusy(true);
  try {
    const resp = await fetch(`${API_BASE}/api/v1/resume`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: state.threadId, action }),
    });
    const data = await resp.json().catch(() => null);
    if (!resp.ok || !data) {
      throw new Error((data && data.detail) || `审批请求失败（HTTP ${resp.status}）`);
    }
    if (data.code === 202) {
      // 流程仍需要下一轮审批，更新提示文案
      desc.textContent = data.interrupt_msg || "Agent 正在尝试执行敏感操作，等待人工授权。";
      setBusy(false);
    } else if (data.code === 200 && data.answer) {
      card.remove();
      state.pendingReview = false;
      finalizeTurn(turn);
      renderAnswer(turn, data.answer);
    } else {
      throw new Error(data.detail || "审批后未返回有效结果");
    }
  } catch (e) {
    toast(e.message || "审批请求失败", "err");
    setBusy(false);
  }
}

/* ==========================================================================
   发送流程
   ========================================================================== */
function setComposerEnabled(enabled) {
  sendBtn.disabled = !enabled;
  inputEl.disabled = !enabled;
}

function removeWelcome() {
  const w = messagesEl.querySelector(".welcome");
  if (w) w.remove();
}

async function send(question) {
  const q = (question ?? inputEl.value).trim();
  if (!q) {
    toast("请输入问题", "err");
    inputEl.focus();
    return;
  }
  const uid = uidInput.value.trim();
  if (!uid) {
    toast("请先在左侧填写员工 UID", "err");
    uidInput.focus();
    openSidebar();
    return;
  }
  if (state.streaming) return;
  if (state.pendingReview) {
    toast("当前会话存在待审批事项，请先完成审批", "err");
    return;
  }

  removeWelcome();
  appendUser(q);
  inputEl.value = "";
  autosize();
  closeSidebar();

  state.streaming = true;
  setComposerEnabled(false);

  const turn = createAssistantTurn();
  showTyping(turn);

  try {
    await streamChat(
      { uid, thread_id: state.threadId, question: q },
      turn
    );
    // SSE 流未产出答案：可能触发了人工审批（interrupt），探测状态并渲染审批卡片
    if (!turn.answered) {
      await probeAfterStream(turn, { uid, thread_id: state.threadId, message: q });
    }
  } catch (e) {
    showTurnError(turn, e.message || "未知错误");
    finalizeTurn(turn);
  } finally {
    state.streaming = false;
    setComposerEnabled(true);
    inputEl.focus();
  }
}

/* ==========================================================================
   健康检查
   ========================================================================== */
function setHealth(status) {
  healthEl.classList.remove("health--ok", "health--err");
  if (status === "ok") {
    healthEl.classList.add("health--ok");
    healthEl.querySelector(".health__text").textContent = "服务在线";
  } else if (status === "err") {
    healthEl.classList.add("health--err");
    healthEl.querySelector(".health__text").textContent = "服务不可用";
  } else {
    healthEl.querySelector(".health__text").textContent = "服务检测中…";
  }
}

async function checkHealth() {
  setHealth("");
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 4000);
    const resp = await fetch(`${API_BASE}/health`, { signal: ctrl.signal });
    clearTimeout(timer);
    const data = await resp.json();
    setHealth(resp.ok && data.status === "ok" ? "ok" : "err");
  } catch {
    setHealth("err");
  }
}

/* ==========================================================================
   会话 / 侧边栏 / 输入
   ========================================================================== */
function resetThread() {
  state.threadId = randomId();
  state.pendingReview = false;
  threadIdEl.textContent = state.threadId;
  messagesEl.innerHTML = "";
  renderWelcome();
  toast("已开启新会话");
}

function openSidebar() {
  sidebarEl.classList.add("sidebar--open");
  scrimEl.hidden = false;
  requestAnimationFrame(() => scrimEl.classList.add("scrim--show"));
  menuBtn.setAttribute("aria-expanded", "true");
}

function closeSidebar() {
  sidebarEl.classList.remove("sidebar--open");
  scrimEl.classList.remove("scrim--show");
  menuBtn.setAttribute("aria-expanded", "false");
  setTimeout(() => (scrimEl.hidden = true), 280);
}

/* 输入框自适应高度 */
function autosize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
}

/* ==========================================================================
   欢迎页
   ========================================================================== */
function renderWelcome() {
  const w = el("div", "welcome");
  w.innerHTML = `
    <span class="welcome__mark">羽</span>
    <h3 class="welcome__title">你好，我是你的 HR 助理</h3>
    <p class="welcome__desc">
      我可以帮你查询年假与病假余额、开具在职 / 收入证明、解读差旅报销与考勤制度，
      并基于你的职级与工作地给出个性化答复。
    </p>
  `;
  messagesEl.appendChild(w);
}

/* ==========================================================================
   初始化
   ========================================================================== */
function init() {
  threadIdEl.textContent = state.threadId;
  renderWelcome();

  // 恢复上次填写的 UID
  const savedUid = localStorage.getItem("hr_agent_uid");
  if (savedUid) uidInput.value = savedUid;

  uidInput.addEventListener("change", () => {
    localStorage.setItem("hr_agent_uid", uidInput.value.trim());
  });

  // 提交
  formEl.addEventListener("submit", (e) => {
    e.preventDefault();
    send();
  });

  // Enter 发送 / Shift+Enter 换行
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    }
  });
  inputEl.addEventListener("input", autosize);

  // 快捷提问
  samplesEl.addEventListener("click", (e) => {
    const btn = e.target.closest(".sample");
    if (btn) send(btn.dataset.q);
  });

  // 会话 / 清空
  newThreadBtn.addEventListener("click", resetThread);
  clearBtn.addEventListener("click", resetThread);

  // 移动端抽屉
  menuBtn.addEventListener("click", () =>
    sidebarEl.classList.contains("sidebar--open") ? closeSidebar() : openSidebar()
  );
  scrimEl.addEventListener("click", closeSidebar);

  // 健康检查（点击重试）
  healthEl.addEventListener("click", checkHealth);
  checkHealth();
  setInterval(checkHealth, 30000);
}

document.addEventListener("DOMContentLoaded", init);

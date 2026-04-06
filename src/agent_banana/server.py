from __future__ import annotations
import os

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Callable

from .pipeline import AgentBananaApp
from .vision import decode_image_payload

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Moleculyst Studio</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0d1117; --bg-sidebar: #161b22; --bg-chat: #0d1117;
      --bg-user: #1f6feb; --bg-agent: #21262d; --bg-error: #3d1f1f;
      --bg-tool: rgba(88,166,255,0.08); --bg-input: #161b22;
      --border: #30363d; --border-hi: #484f58;
      --text: #e6edf3; --text-dim: #8b949e; --text-muted: #6e7681;
      --accent: #58a6ff; --green: #3fb950; --amber: #d29922;
      --red: #f85149; --purple: #bc8cff; --orange: #f0883e;
      --radius: 12px; --radius-sm: 8px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { height: 100vh; font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); display: flex; overflow: hidden; }

    /* ── Sidebar ── */
    .sidebar {
      width: 260px; background: var(--bg-sidebar); border-right: 1px solid var(--border);
      display: flex; flex-direction: column; flex-shrink: 0;
    }
    .sidebar-header {
      padding: 16px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; gap: 10px;
    }
    .logo { width: 32px; height: 32px; border-radius: 8px; background: linear-gradient(135deg, var(--amber), var(--orange)); display: flex; align-items: center; justify-content: center; font-size: 18px; }
    .sidebar-header h1 { font-size: 1rem; font-weight: 700; letter-spacing: -0.01em; }
    .sidebar-section { padding: 12px 16px; }
    .sidebar-section h3 { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 8px; font-weight: 600; }
    .upload-btn {
      width: 100%; padding: 10px; border: 1px dashed var(--border-hi); border-radius: var(--radius-sm);
      background: transparent; color: var(--text-dim); cursor: pointer; font-size: 0.82rem;
      transition: all 0.2s; text-align: center;
    }
    .upload-btn:hover { border-color: var(--accent); color: var(--accent); }
    .image-list { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
    .image-thumb {
      width: 100%; border-radius: var(--radius-sm); object-fit: contain; max-height: 180px;
      border: 2px solid transparent; cursor: pointer; transition: border 0.2s;
    }
    .image-thumb.active { border-color: var(--accent); }
    .image-thumb:hover { border-color: var(--accent); opacity: 0.9; }
    .sidebar-tools { flex: 1; overflow-y: auto; padding: 12px 16px; }
    .sidebar-tools h3 { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 8px; font-weight: 600; }
    .tool-item {
      padding: 6px 8px; border-radius: 6px; font-size: 0.75rem; color: var(--text-dim);
      font-family: 'JetBrains Mono', monospace; margin-bottom: 2px;
    }
    .tool-item:hover { background: rgba(255,255,255,0.04); }

    /* ── Main area ── */
    .main { flex: 1; display: flex; flex-direction: column; min-width: 0; }

    /* Top bar */
    .topbar {
      padding: 10px 20px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between;
      background: var(--bg-sidebar);
    }
    .topbar-title { font-size: 0.85rem; color: var(--text-dim); font-weight: 500; }
    .stop-btn {
      padding: 6px 16px; border-radius: 6px; border: 1px solid var(--red);
      background: rgba(248,81,73,0.1); color: var(--red); font: 600 0.78rem 'Inter', sans-serif;
      cursor: pointer; transition: all 0.2s; display: none;
    }
    .stop-btn:hover { background: rgba(248,81,73,0.2); }
    .stop-btn.visible { display: inline-flex; align-items: center; gap: 6px; }

    /* Chat area */
    .chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
    .chat::-webkit-scrollbar { width: 6px; }
    .chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* Chat bubbles */
    .msg { display: flex; gap: 10px; max-width: 85%; animation: fadeIn 0.3s ease; }
    .msg.user { align-self: flex-end; flex-direction: row-reverse; }
    .msg.agent { align-self: flex-start; }

    .msg-avatar {
      width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0;
      display: flex; align-items: center; justify-content: center; font-size: 14px;
    }
    .msg.user .msg-avatar { background: var(--bg-user); }
    .msg.agent .msg-avatar { background: var(--border); }

    .msg-body { display: flex; flex-direction: column; gap: 6px; }
    .msg-sender { font-size: 0.7rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }

    .msg-bubble {
      padding: 12px 16px; border-radius: var(--radius); font-size: 0.88rem; line-height: 1.55;
      word-break: break-word;
    }
    .msg.user .msg-bubble { background: var(--bg-user); border-bottom-right-radius: 4px; }
    .msg.agent .msg-bubble { background: var(--bg-agent); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
    .msg.agent .msg-bubble.tool-call { background: var(--bg-tool); border-color: rgba(88,166,255,0.2); }
    .msg.agent .msg-bubble.error { background: var(--bg-error); border-color: rgba(248,81,73,0.3); }

    .tool-badge {
      display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px;
      background: rgba(88,166,255,0.15); border-radius: 4px; font-size: 0.72rem;
      font-family: 'JetBrains Mono', monospace; color: var(--accent); font-weight: 500;
    }

    /* Images in chat */
    .chat-images { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
    .chat-images img {
      max-width: 300px; max-height: 250px; border-radius: var(--radius-sm);
      border: 1px solid var(--border); cursor: pointer; transition: transform 0.2s;
    }
    .chat-images img:hover { transform: scale(1.02); }
    .img-label { font-size: 0.7rem; color: var(--text-muted); text-align: center; margin-top: 2px; }

    /* Quality metrics in chat */
    .chat-metrics {
      display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px;
      padding: 8px 12px; background: rgba(0,0,0,0.2); border-radius: var(--radius-sm);
    }
    .metric { text-align: center; }
    .metric .label { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; }
    .metric .value { font-size: 0.85rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

    /* BBox editor in chat */
    .bbox-editor-card {
      background: var(--bg-agent); border: 1px solid var(--border); border-radius: var(--radius);
      padding: 16px; margin-top: 8px; max-width: 500px;
    }
    .bbox-editor-card h4 { font-size: 0.78rem; color: var(--text-muted); font-weight: 600; margin-bottom: 8px; }
    .bbox-editor-wrap { position: relative; display: inline-block; }
    .bbox-editor-wrap img { max-width: 100%; border-radius: var(--radius-sm); display: block; }
    .bbox-editor-wrap canvas { position: absolute; top: 0; left: 0; cursor: crosshair; }
    .bbox-controls { display: flex; gap: 8px; align-items: center; margin-top: 8px; flex-wrap: wrap; }
    .bbox-coords {
      font: 500 0.75rem 'JetBrains Mono', monospace; color: var(--text-dim);
      padding: 4px 8px; background: rgba(0,0,0,0.3); border-radius: 4px;
    }
    .bbox-input {
      flex: 1; min-width: 180px; padding: 6px 10px; border-radius: 6px;
      border: 1px solid var(--border); background: rgba(0,0,0,0.3);
      color: var(--text); font-size: 0.82rem;
    }
    .bbox-input:focus { border-color: var(--accent); outline: none; }

    .btn {
      border: 0; border-radius: 6px; padding: 8px 16px; cursor: pointer;
      font: 600 0.82rem 'Inter', sans-serif; transition: all 0.2s; color: #fff;
    }
    .btn-green { background: linear-gradient(135deg, var(--green), #2ea043); }
    .btn-green:hover { transform: translateY(-1px); }
    .btn-green:disabled { opacity: 0.5; cursor: wait; transform: none; }
    .btn-blue { background: linear-gradient(135deg, var(--accent), #388bfd); }
    .btn-red { background: linear-gradient(135deg, var(--red), #da3633); }

    /* Input bar */
    .input-bar {
      padding: 12px 20px; border-top: 1px solid var(--border);
      background: var(--bg-input); display: flex; gap: 10px; align-items: end;
    }
    .input-bar-file {
      width: 36px; height: 36px; border-radius: 8px; border: 1px solid var(--border);
      background: transparent; color: var(--text-dim); cursor: pointer;
      display: flex; align-items: center; justify-content: center; font-size: 18px;
      transition: all 0.2s; flex-shrink: 0;
    }
    .input-bar-file:hover { border-color: var(--accent); color: var(--accent); }
    .input-bar textarea {
      flex: 1; border: 1px solid var(--border); border-radius: var(--radius-sm);
      background: rgba(0,0,0,0.3); padding: 8px 12px; color: var(--text);
      font: 0.88rem/1.4 'Inter', sans-serif; resize: none; height: 36px; max-height: 120px;
      outline: none; transition: border 0.2s;
    }
    .input-bar textarea:focus { border-color: var(--accent); }
    .input-bar .send-btn {
      width: 36px; height: 36px; border-radius: 8px; border: 0;
      background: var(--accent); color: #fff; cursor: pointer;
      display: flex; align-items: center; justify-content: center; font-size: 16px;
      transition: all 0.2s; flex-shrink: 0;
    }
    .input-bar .send-btn:hover { background: #388bfd; }
    .input-bar .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

    /* Typing indicator */
    .typing { display: inline-flex; gap: 4px; padding: 8px 12px; }
    .typing span {
      width: 6px; height: 6px; background: var(--text-muted); border-radius: 50%;
      animation: bounce 1.4s infinite ease-in-out;
    }
    .typing span:nth-child(1) { animation-delay: 0s; }
    .typing span:nth-child(2) { animation-delay: 0.2s; }
    .typing span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

    /* Welcome */
    .welcome {
      flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
      color: var(--text-dim); gap: 12px;
    }
    .welcome-icon { font-size: 48px; }
    .welcome h2 { font-size: 1.2rem; font-weight: 700; color: var(--text); }
    .welcome p { font-size: 0.88rem; max-width: 400px; text-align: center; line-height: 1.5; }

    /* Hide file input */
    input[type="file"] { display: none; }

    /* Mode toggle */
    .mode-toggle-section {
      padding: 12px 16px; border-bottom: 1px solid var(--border);
    }
    .mode-toggle-row {
      display: flex; align-items: center; justify-content: space-between;
      gap: 8px;
    }
    .mode-label {
      font-size: 0.75rem; font-weight: 600; color: var(--text-dim);
      text-transform: uppercase; letter-spacing: 0.05em;
    }
    .mode-label.active { color: var(--accent); }
    .toggle-switch {
      position: relative; width: 44px; height: 24px;
      background: var(--border); border-radius: 12px;
      cursor: pointer; transition: background 0.3s; flex-shrink: 0;
    }
    .toggle-switch.active { background: var(--accent); }
    .toggle-switch .toggle-knob {
      position: absolute; top: 3px; left: 3px;
      width: 18px; height: 18px; background: var(--text);
      border-radius: 50%; transition: transform 0.3s;
    }
    .toggle-switch.active .toggle-knob { transform: translateX(20px); }
    .mode-description {
      font-size: 0.7rem; color: var(--text-muted); margin-top: 6px; line-height: 1.4;
    }

    /* Manual mode confirmation cards */
    .manual-confirm-card {
      background: var(--bg-agent); border: 1px solid var(--accent);
      border-radius: var(--radius); padding: 16px; margin-top: 8px; max-width: 500px;
    }
    .manual-confirm-card h4 {
      font-size: 0.82rem; color: var(--accent); font-weight: 600; margin-bottom: 8px;
    }
    .confirm-btn-row { display: flex; gap: 8px; margin-top: 10px; }
  </style>
</head>
<body>
  <!-- Sidebar -->
  <div class="sidebar">
    <div class="sidebar-header">
      <img src="data:image/jpeg;base64,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" class="logo" />
      <h1>Moleculyst</h1>
    </div>
    <div class="sidebar-section">
      <h3>Images</h3>
      <label class="upload-btn" for="sidebarFileInput">📎 Upload Image</label>
      <input type="file" id="sidebarFileInput" accept="image/*">
      <div class="image-list" id="imageList"></div>
    </div>
    <div class="mode-toggle-section">
      <div class="mode-toggle-row">
        <span class="mode-label active" id="autoLabel">Auto</span>
        <div class="toggle-switch" id="modeToggle" onclick="toggleMode()">
          <div class="toggle-knob"></div>
        </div>
        <span class="mode-label" id="manualLabel">Manual</span>
      </div>
      <div class="mode-description" id="modeDesc">Agent runs automatically — grounding, editing, and quality checks happen without interruption.</div>
    </div>
    <div class="sidebar-tools">
    </div>
  </div>

  <!-- Main -->
  <div class="main">
    <div class="topbar">
      <span class="topbar-title" id="sessionLabel">New Session</span>
      <button class="stop-btn" id="stopBtn" onclick="handleStop()">■ Stop</button>
    </div>

    <div class="chat" id="chat">
      <div class="welcome" id="welcome">
        <img src="data:image/jpeg;base64,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" style="width: 64px; height: 64px; border-radius: 8px; object-fit: contain;">
        <h2>Moleculyst Studio</h2>
        <p>Upload an image and describe what you'd like to edit. I'll show you my reasoning at every step.</p>
      </div>
    </div>

    <div class="input-bar">
      <label class="input-bar-file" for="inputFile" title="Upload image">📎</label>
      <input type="file" id="inputFile" accept="image/*">
      <textarea id="userInput" placeholder="Describe your edit..." rows="1"
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();handleSend();}"></textarea>
      <button class="send-btn" id="sendBtn" onclick="handleSend()" title="Send">➤</button>
    </div>
  </div>

<script>
  var chat = document.getElementById("chat");
  var welcome = document.getElementById("welcome");
  var userInput = document.getElementById("userInput");
  var sendBtn = document.getElementById("sendBtn");
  var stopBtn = document.getElementById("stopBtn");
  var inputFile = document.getElementById("inputFile");
  var sidebarFile = document.getElementById("sidebarFileInput");
  var imageList = document.getElementById("imageList");

  var _sourcePayload = null;
  var _images = [];
  var _running = false;
  var _iterCount = 0;
  var _aborted = false;
  var _mode = "auto";  // "auto" or "manual"

  function toggleMode() {
    var toggle = document.getElementById("modeToggle");
    var autoLabel = document.getElementById("autoLabel");
    var manualLabel = document.getElementById("manualLabel");
    var modeDesc = document.getElementById("modeDesc");
    if (_mode === "auto") {
      _mode = "manual";
      toggle.classList.add("active");
      autoLabel.classList.remove("active");
      manualLabel.classList.add("active");
      modeDesc.textContent = "You confirm or adjust the bounding box for each target before the edit runs.";
    } else {
      _mode = "auto";
      toggle.classList.remove("active");
      autoLabel.classList.add("active");
      manualLabel.classList.remove("active");
      modeDesc.textContent = "Agent runs automatically \u2014 grounding, editing, and quality checks happen without interruption.";
    }
  }

  function esc(s) { var d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

  function scrollToBottom() { chat.scrollTop = chat.scrollHeight; }

  function addUserMsg(text, imgSrc) {
    welcome.style.display = "none";
    var html = '<div class="msg user"><div class="msg-avatar">👤</div><div class="msg-body">' +
      '<div class="msg-sender">You</div><div class="msg-bubble">' + esc(text) + '</div>';
    if (imgSrc) {
      html += '<div class="chat-images"><div><img src="' + imgSrc + '" style="max-width:200px"><div class="img-label">Uploaded</div></div></div>';
    }
    html += '</div></div>';
    chat.insertAdjacentHTML("beforeend", html);
    scrollToBottom();
  }

  function addAgentMsg(text, extra) {
    var html = '<div class="msg agent"><div class="msg-avatar">⚛️</div><div class="msg-body">' +
      '<div class="msg-sender">Moleculyst</div><div class="msg-bubble">' + text + '</div>';
    if (extra) html += extra;
    html += '</div></div>';
    chat.insertAdjacentHTML("beforeend", html);
    scrollToBottom();
    return chat.lastElementChild;
  }

  function addToolMsg(toolName, thought, detail) {
    var html = '<div class="msg agent"><div class="msg-avatar">🛠</div><div class="msg-body">' +
      '<div class="msg-sender">Tool Call</div>' +
      '<div class="msg-bubble tool-call">' +
      '<div><span class="tool-badge">⚡ ' + esc(toolName) + '</span></div>' +
      '<div style="margin-top:6px;color:var(--text-dim);font-size:0.82rem">💭 ' + esc(thought) + '</div>';
    if (detail) html += '<div style="margin-top:6px">' + detail + '</div>';
    html += '</div></div></div>';
    chat.insertAdjacentHTML("beforeend", html);
    scrollToBottom();
  }

  function addErrorMsg(text) {
    addAgentMsg('<span style="color:var(--red)">⚠ ' + esc(text) + '</span>');
  }

  function addTyping() {
    var html = '<div class="msg agent" id="typingIndicator"><div class="msg-avatar">⚛️</div><div class="msg-body"><div class="typing"><span></span><span></span><span></span></div></div></div>';
    chat.insertAdjacentHTML("beforeend", html);
    scrollToBottom();
  }
  function removeTyping() { var el = document.getElementById("typingIndicator"); if (el) el.remove(); }

  function addImageComparison(beforeSrc, afterSrc, beforeLabel, afterLabel) {
    return '<div class="chat-images">' +
      '<div><img src="' + beforeSrc + '"><div class="img-label">' + (beforeLabel || "Before") + '</div></div>' +
      '<div><img src="' + afterSrc + '"><div class="img-label">' + (afterLabel || "After") + '</div></div>' +
      '</div>';
  }

  function addMetrics(q) {
    var seamColor = q.seam_verdict === "accept" ? "var(--green)" : q.seam_verdict === "warn" ? "var(--amber)" : "var(--red)";
    var semColor = (q.semantic_score||1) >= 0.7 ? "var(--green)" : (q.semantic_score||1) >= 0.4 ? "var(--amber)" : "var(--red)";
    var html = '<div class="chat-metrics">' +
      '<div class="metric"><div class="label">Score</div><div class="value">' + (q.score||0).toFixed(3) + '</div></div>' +
      '<div class="metric"><div class="label">Inside Δ</div><div class="value">' + (q.inside_change||0).toFixed(3) + '</div></div>' +
      '<div class="metric"><div class="label">Outside Δ</div><div class="value">' + (q.outside_change||0).toFixed(3) + '</div></div>' +
      '<div class="metric"><div class="label">Seam</div><div class="value" style="color:' + seamColor + '">' + (q.seam_score||0).toFixed(3) + ' ' + (q.seam_verdict||"") + '</div></div>' +
      '<div class="metric"><div class="label">Semantic</div><div class="value" style="color:' + semColor + '">' + (q.semantic_score != null ? (q.semantic_score).toFixed(2) : "—") + ' ' + (q.semantic_fulfilled ? "✓" : "✗") + '</div></div>' +
      '<div class="metric"><div class="label">Status</div><div class="value" style="color:' + (q.accepted ? "var(--green)" : "var(--red)") + '">' + (q.accepted ? "✓ Pass" : "✗ Fail") + '</div></div>' +
      '</div>';
    if (q.semantic_reasoning) {
      html += '<div style="margin-top:6px;padding:8px 12px;background:rgba(0,0,0,0.2);border-radius:8px;font-size:0.8rem;color:var(--text-dim)">🧠 <strong>VLM Critic:</strong> ' + esc(q.semantic_reasoning) + '</div>';
    }
    return html;
  }

  function setRunning(v) {
    _running = v;
    sendBtn.disabled = v;
    stopBtn.classList.toggle("visible", v);
  }

  function handleStop() {
    _aborted = true;
    setRunning(false);
    removeTyping();
    addAgentMsg("🛑 Stopped by user. You can give me new instructions.");
  }

  function dataUrlFromFile(file) {
    return new Promise(function(resolve, reject) {
      var reader = new FileReader();
      reader.onload = function() { resolve(reader.result); };
      reader.onerror = function() { reject(new Error("Failed to read")); };
      reader.readAsDataURL(file);
    });
  }

  function addToSidebar(dataUrl) {
    _images.push(dataUrl);
    var img = document.createElement("img");
    img.src = dataUrl;
    img.className = "image-thumb active";
    img.onclick = function() {
      _sourcePayload = dataUrl;
      document.querySelectorAll(".image-thumb").forEach(function(el) { el.classList.remove("active"); });
      img.classList.add("active");
      welcome.style.display = "none";
      addAgentMsg("🖼 Switched active image. Your next instruction will edit this image.", '<div class="chat-images"><div><img src="' + dataUrl + '" style="max-width:400px"><div class="img-label">Active Image</div></div></div>');
    };
    document.querySelectorAll(".image-thumb").forEach(function(el) { el.classList.remove("active"); });
    imageList.appendChild(img);
  }

  // File upload handlers
  inputFile.addEventListener("change", function(e) {
    if (e.target.files[0]) {
      dataUrlFromFile(e.target.files[0]).then(function(url) {
        _sourcePayload = url;
        addToSidebar(url);
        welcome.style.display = "none";
        addAgentMsg("📷 Image uploaded and ready for editing.", '<div class="chat-images"><div><img src="' + url + '" style="max-width:400px"><div class="img-label">Source Image</div></div></div>');
      });
    }
  });
  sidebarFile.addEventListener("change", function(e) {
    if (e.target.files[0]) {
      dataUrlFromFile(e.target.files[0]).then(function(url) {
        _sourcePayload = url;
        addToSidebar(url);
        welcome.style.display = "none";
        addAgentMsg("📷 Image uploaded and ready for editing.", '<div class="chat-images"><div><img src="' + url + '" style="max-width:400px"><div class="img-label">Source Image</div></div></div>');
      });
    }
  });

  // Auto-resize textarea
  userInput.addEventListener("input", function() {
    this.style.height = "36px";
    this.style.height = Math.min(this.scrollHeight, 120) + "px";
  });

  function renderAgentStep(step) {
    var actionIcon = {
      "expand_region": "\ud83d\udcd0", "crop_local_patch": "\u2702\ufe0f",
      "edit_local": "\ud83c\udfa8", "blend_back": "\ud83d\udd17",
      "evaluate_quality": "\ud83d\udcca", "verify_semantic": "\ud83e\udde0",
      "adjust_strategy": "\ud83d\udd27", "return_best": "\ud83c\udfc1", "finish": "\u2705"
    }[step.action] || "\u2699\ufe0f";

    var detail = '<div style="font-size:0.82rem;color:var(--text-dim);margin-top:2px">\ud83d\udcad ' + esc(step.thought) + '</div>';
    detail += '<div style="font-size:0.82rem;margin-top:2px">\ud83d\udc41 ' + esc(step.observation) + '</div>';

    if (step.critic_verdict) {
      var cv = step.critic_verdict;
      var cvColor = cv.fulfilled ? "var(--green)" : "var(--red)";
      detail += '<div style="margin-top:4px;padding:6px 10px;background:rgba(0,0,0,0.2);border-radius:6px;border-left:3px solid ' + cvColor + '">';
      detail += '<strong style="color:' + cvColor + '">' + (cv.fulfilled ? "\u2713 Approved" : "\u2717 Rejected") + '</strong>';
      detail += ' (score: ' + (cv.semantic_score||0).toFixed(2) + ')';
      detail += '<div style="font-size:0.78rem;color:var(--text-dim);margin-top:2px">' + esc(cv.reasoning) + '</div>';
      if (cv.residual_objects && cv.residual_objects.length) {
        detail += '<div style="font-size:0.78rem;color:var(--red);margin-top:2px">Still visible: ' + cv.residual_objects.map(esc).join(", ") + '</div>';
      }
      if (cv.suggestions && cv.suggestions.length) {
        detail += '<div style="font-size:0.78rem;color:var(--amber);margin-top:2px">\ud83d\udca1 ' + cv.suggestions.map(esc).join("; ") + '</div>';
      }
      detail += '</div>';
    }

    if (step.image_url) {
      detail += '<div style="margin-top:6px"><img src="' + step.image_url + '" style="max-width:300px;border-radius:8px;border:1px solid var(--border)"></div>';
    }

    var durationStr = step.duration_ms ? " (" + (step.duration_ms/1000).toFixed(1) + "s)" : "";
    addToolMsg(step.action, actionIcon + " " + step.action + durationStr, detail);
  }

  function handleSend() {
    if (_mode === "manual") {
      handleManualSend();
    } else {
      handleAutoSend();
    }
  }

  function handleAutoSend() {
    var text = userInput.value.trim();
    if (!text || _running) return;
    if (!_sourcePayload) {
      addAgentMsg("Please upload an image first using the 📎 button.");
      return;
    }
    userInput.value = "";
    userInput.style.height = "36px";
    _aborted = false;

    addUserMsg(text);
    setRunning(true);
    addTyping();

    // Use SSE streaming endpoint for real-time step rendering
    fetch("/api/edit-stream", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ image: _sourcePayload, instruction: text })
    })
    .then(function(response) {
      if (!response.ok) {
        return response.json().then(function(err) { throw new Error(err.error || "Request failed"); });
      }
      removeTyping();
      return readSSEStream(response, text);
    })
    .catch(function(err) {
      removeTyping();
      addErrorMsg(err.message);
      setRunning(false);
    });
  }

  function handleManualSend() {
    var text = userInput.value.trim();
    if (!text || _running) return;
    if (!_sourcePayload) {
      addAgentMsg("Please upload an image first using the 📎 button.");
      return;
    }
    userInput.value = "";
    userInput.style.height = "36px";
    _aborted = false;

    addUserMsg(text);
    setRunning(true);
    addTyping();

    // Step 1: Ground targets — find bboxes for each target
    fetch("/api/ground", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ image: _sourcePayload, instruction: text })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      removeTyping();
      if (_aborted) { setRunning(false); return; }
      if (data.error) { addErrorMsg(data.error); setRunning(false); return; }

      var targets = data.targets || [];
      if (targets.length === 0) {
        addErrorMsg("No targets found in instruction.");
        setRunning(false);
        return;
      }

      addAgentMsg("🔍 <strong>Manual Mode</strong> — I found " + targets.length + " target(s). Please review and adjust the bounding boxes below, then click <strong>Confirm & Edit</strong>.");

      // Create bbox editors for each target, store references
      var confirmedBboxes = [];
      var editorsContainer = document.createElement("div");
      editorsContainer.style.cssText = "display:flex;flex-direction:column;gap:12px;";

      targets.forEach(function(t, idx) {
        confirmedBboxes.push(Object.assign({}, t.bbox));

        var card = document.createElement("div");
        card.className = "manual-confirm-card";
        card.innerHTML =
          '<h4>Target ' + (idx+1) + ': ' + esc(t.target) + '</h4>' +
          '<div style="font-size:0.8rem;color:var(--text-dim);margin-bottom:6px">' + esc(t.prompt) + '</div>' +
          '<div class="bbox-editor-wrap" id="manual-wrap-' + idx + '">' +
            '<img id="manual-img-' + idx + '" src="' + t.overlay_data_url + '" style="max-width:100%;border-radius:8px;display:block">' +
            '<canvas id="manual-cvs-' + idx + '" style="position:absolute;top:0;left:0;cursor:crosshair"></canvas>' +
          '</div>' +
          '<div style="margin-top:6px">' +
            '<span class="bbox-coords" id="manual-coords-' + idx + '">[' + t.bbox.left + ',' + t.bbox.top + '] → [' + t.bbox.right + ',' + t.bbox.bottom + '] (' + (t.bbox.right-t.bbox.left) + '×' + (t.bbox.bottom-t.bbox.top) + 'px)</span>' +
          '</div>';
        editorsContainer.appendChild(card);

        // Wire up bbox drawing on this card's canvas
        (function(tIdx, tData) {
          setTimeout(function() {
            var img = document.getElementById("manual-img-" + tIdx);
            var cvs = document.getElementById("manual-cvs-" + tIdx);
            var coords = document.getElementById("manual-coords-" + tIdx);
            if (!img || !cvs) return;

            cvs.width = img.offsetWidth;
            cvs.height = img.offsetHeight;
            var ctx = cvs.getContext("2d");
            var scaleX = img.offsetWidth / tData.image_width;
            var scaleY = img.offsetHeight / tData.image_height;

            // Draw initial bbox
            ctx.strokeStyle = "rgba(88,166,255,0.8)";
            ctx.lineWidth = 2;
            ctx.setLineDash([6,3]);
            var bx = tData.bbox;
            ctx.strokeRect(bx.left*scaleX, bx.top*scaleY, (bx.right-bx.left)*scaleX, (bx.bottom-bx.top)*scaleY);

            var drawing = false, startX, startY;
            cvs.addEventListener("mousedown", function(e) {
              drawing = true;
              var rect = cvs.getBoundingClientRect();
              startX = e.clientX - rect.left;
              startY = e.clientY - rect.top;
            });
            cvs.addEventListener("mousemove", function(e) {
              if (!drawing) return;
              var rect = cvs.getBoundingClientRect();
              var cx = e.clientX - rect.left;
              var cy = e.clientY - rect.top;
              ctx.clearRect(0,0,cvs.width,cvs.height);
              ctx.strokeStyle = "rgba(88,166,255,0.8)";
              ctx.lineWidth = 2;
              ctx.setLineDash([6,3]);
              ctx.strokeRect(startX, startY, cx-startX, cy-startY);
            });
            cvs.addEventListener("mouseup", function(e) {
              if (!drawing) return;
              drawing = false;
              var rect = cvs.getBoundingClientRect();
              var cx = e.clientX - rect.left;
              var cy = e.clientY - rect.top;
              var l = Math.round(Math.min(startX, cx) / scaleX);
              var t2 = Math.round(Math.min(startY, cy) / scaleY);
              var r = Math.round(Math.max(startX, cx) / scaleX);
              var b = Math.round(Math.max(startY, cy) / scaleY);
              l = Math.max(0, l); t2 = Math.max(0, t2);
              r = Math.min(tData.image_width, r); b = Math.min(tData.image_height, b);
              if (r - l > 5 && b - t2 > 5) {
                confirmedBboxes[tIdx] = {left: l, top: t2, right: r, bottom: b};
                coords.textContent = "[" + l + "," + t2 + "] → [" + r + "," + b + "] (" + (r-l) + "×" + (b-t2) + "px)";
              }
            });
          }, 100);
        })(idx, t);
      });

      // Add confirm/cancel buttons via DOM API (no hardcoded IDs)
      var btnRow = document.createElement("div");
      btnRow.className = "confirm-btn-row";
      var confirmBtn = document.createElement("button");
      confirmBtn.className = "btn btn-green";
      confirmBtn.textContent = "Confirm & Edit";
      var cancelBtn = document.createElement("button");
      cancelBtn.className = "btn btn-red";
      cancelBtn.textContent = "Cancel";
      btnRow.appendChild(confirmBtn);
      btnRow.appendChild(cancelBtn);
      editorsContainer.appendChild(btnRow);

      var msgDiv = document.createElement("div");
      msgDiv.className = "msg agent";
      var msgBody = document.createElement("div");
      msgBody.className = "msg-body";
      var msgSender = document.createElement("div");
      msgSender.className = "msg-sender";
      msgSender.textContent = "Moleculyst";
      var msgAvatar = document.createElement("div");
      msgAvatar.className = "msg-avatar";
      msgAvatar.textContent = "⚛️";
      msgBody.appendChild(msgSender);
      msgBody.appendChild(editorsContainer);
      msgDiv.appendChild(msgAvatar);
      msgDiv.appendChild(msgBody);
      chat.appendChild(msgDiv);
      scrollToBottom();

      // Wire confirm button directly
      confirmBtn.addEventListener("click", function() {
        confirmBtn.disabled = true;
        confirmBtn.textContent = "Editing...";
        cancelBtn.disabled = true;
        addTyping();
        fetch("/api/edit-manual-stream", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            image: _sourcePayload,
            instruction: text,
            bboxes: confirmedBboxes
          })
        })
        .then(function(response) {
          if (!response.ok) {
            return response.json().then(function(err) { throw new Error(err.error || "Request failed"); });
          }
          removeTyping();
          return readSSEStream(response, text);
        })
        .catch(function(err) {
          removeTyping();
          addErrorMsg(err.message);
          setRunning(false);
        });
      });

      cancelBtn.addEventListener("click", function() {
        addAgentMsg("Edit cancelled.");
        confirmBtn.disabled = true;
        cancelBtn.disabled = true;
        setRunning(false);
      });
    })
    .catch(function(err) {
      removeTyping();
      addErrorMsg(err.message);
      setRunning(false);
    });
  }

  // Shared SSE stream reader used by both auto and manual modes
  function readSSEStream(response, instructionText) {
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    var eventType = "message";  // Persists across processStream() calls

    function processStream() {
      return reader.read().then(function(result) {
        if (result.done || _aborted) {
          setRunning(false);
          return;
        }

        buffer += decoder.decode(result.value, {stream: true});
        var lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];
          if (line.startsWith("event: ")) {
            eventType = line.substring(7).trim();
          } else if (line.startsWith("data: ")) {
            var jsonStr = line.substring(6);
            try {
              var data = JSON.parse(jsonStr);

              if (eventType === "error") {
                addErrorMsg(data.error || "Unknown error");
                setRunning(false);
                return;
              }

              if (eventType === "done") {
                var steps = data.step_results || [];
                steps.forEach(function(sr, idx) {
                  var groundingText = "I'm looking for <strong>" + esc(sr.step.target) + "</strong>";
                  if (sr.grounding_phrases && sr.grounding_phrases.length) {
                    groundingText += " using phrases: " + sr.grounding_phrases.map(function(p) { return '"' + esc(p) + '"'; }).join(", ");
                  }
                  if (sr.llm_confidence > 0) {
                    groundingText += "<br>LLM confidence: <strong>" + sr.llm_confidence.toFixed(2) + "</strong>";
                  }
                  addToolMsg("ground_target", "Finding the target object", groundingText);
                  var metrics = addMetrics(sr.quality);
                  var notes = (sr.quality.notes || []).map(function(n) {
                    return '<div style="font-size:0.8rem;color:var(--text-dim);margin-top:4px">📝 ' + esc(n) + '</div>';
                  }).join("");
                  var attemptsTag = sr.attempts > 1 ? " (" + sr.attempts + " attempts)" : "";
                  addAgentMsg("✅ Step " + (idx+1) + " complete" + attemptsTag + " — <strong>" + esc(sr.step.prompt) + "</strong>", metrics + notes);
                });

                if (data.final_image) {
                  addAgentMsg("🎨 <strong>Here is your edited image:</strong>",
                    '<div class="chat-images"><div><img src="' + data.final_image + '" style="max-width:500px;border-radius:12px"><div class="img-label">Final Result</div></div></div>');
                  _sourcePayload = data.final_image;
                  addToSidebar(data.final_image);
                }

                var lastSr = steps[steps.length - 1];
                if (lastSr && data.final_image) {
                  addAgentMsg("Want to refine? Draw a bounding box below and re-compose:");
                  createBboxEditor(data.final_image, lastSr.bbox, lastSr.image_width || 512, lastSr.image_height || 512, 0, lastSr);
                }
                setRunning(false);
              } else {
                renderAgentStep(data);
              }
            } catch (e) {
              // Ignore parse errors from incomplete JSON
            }
            eventType = "message";
          }
        }
        return processStream();
      });
    }
    return processStream();
  }

  /* ── BBox Editor ── */
  function createBboxEditor(previewSrc, initBbox, imgW, imgH, stepIdx, stepData) {
    var editorId = "bbox-editor-" + stepIdx + "-" + _iterCount;
    var card = document.createElement("div");
    card.className = "msg agent";
    card.innerHTML =
      '<div class="msg-avatar">✏️</div><div class="msg-body"><div class="msg-sender">Adjust</div>' +
      '<div class="bbox-editor-card">' +
        '<h4>Draw a new bounding box and re-compose</h4>' +
        '<div class="bbox-editor-wrap" id="wrap-' + editorId + '">' +
          '<img id="img-' + editorId + '" src="' + previewSrc + '">' +
          '<canvas id="cvs-' + editorId + '"></canvas>' +
        '</div>' +
        '<div class="bbox-controls">' +
          '<span class="bbox-coords" id="coords-' + editorId + '">[' + initBbox.left + ',' + initBbox.top + '] → [' + initBbox.right + ',' + initBbox.bottom + ']</span>' +
          '<input type="text" class="bbox-input" id="instr-' + editorId + '" placeholder="Custom instruction...">' +
          '<button class="btn btn-green" id="recomp-' + editorId + '">Re-compose</button>' +
        '</div>' +
      '</div></div>';
    chat.appendChild(card);
    scrollToBottom();

    setTimeout(function() {
      var img = document.getElementById("img-" + editorId);
      var cvs = document.getElementById("cvs-" + editorId);
      var coordsEl = document.getElementById("coords-" + editorId);
      var recompBtn = document.getElementById("recomp-" + editorId);
      if (!img || !cvs) return;

      function setup() {
        var dispW = img.clientWidth, dispH = img.clientHeight;
        cvs.width = dispW; cvs.height = dispH;
        var scaleX = dispW / imgW, scaleY = dispH / imgH;
        var ctx = cvs.getContext("2d");
        var box = { x1: initBbox.left*scaleX, y1: initBbox.top*scaleY, x2: initBbox.right*scaleX, y2: initBbox.bottom*scaleY };
        var drawing = false;

        function drawBox() {
          ctx.clearRect(0, 0, dispW, dispH);
          ctx.strokeStyle = "#58a6ff"; ctx.lineWidth = 2; ctx.setLineDash([6,3]);
          ctx.strokeRect(box.x1, box.y1, box.x2-box.x1, box.y2-box.y1);
          ctx.fillStyle = "rgba(88,166,255,0.12)";
          ctx.fillRect(box.x1, box.y1, box.x2-box.x1, box.y2-box.y1);
          ctx.fillStyle = "#58a6ff";
          [[box.x1,box.y1],[box.x2,box.y1],[box.x1,box.y2],[box.x2,box.y2]].forEach(function(pt) {
            ctx.beginPath(); ctx.arc(pt[0], pt[1], 4, 0, Math.PI*2); ctx.fill();
          });
          var rb = getRealBox();
          coordsEl.textContent = '[' + rb.left + ',' + rb.top + '] → [' + rb.right + ',' + rb.bottom + '] (' + (rb.right-rb.left) + '×' + (rb.bottom-rb.top) + 'px)';
        }

        function getRealBox() {
          var l=Math.round(Math.min(box.x1,box.x2)/scaleX), t=Math.round(Math.min(box.y1,box.y2)/scaleY);
          var r=Math.round(Math.max(box.x1,box.x2)/scaleX), b=Math.round(Math.max(box.y1,box.y2)/scaleY);
          return { left: Math.max(0,l), top: Math.max(0,t), right: Math.min(imgW,r), bottom: Math.min(imgH,b) };
        }

        cvs.addEventListener("mousedown", function(e) {
          var rect = cvs.getBoundingClientRect();
          box.x1 = e.clientX-rect.left; box.y1 = e.clientY-rect.top;
          box.x2 = box.x1; box.y2 = box.y1; drawing = true;
        });
        cvs.addEventListener("mousemove", function(e) {
          if (!drawing) return;
          var rect = cvs.getBoundingClientRect();
          box.x2 = e.clientX-rect.left; box.y2 = e.clientY-rect.top;
          drawBox();
        });
        cvs.addEventListener("mouseup", function() { drawing = false; });

        recompBtn.addEventListener("click", function() {
          var realBox = getRealBox();
          if (realBox.right-realBox.left < 5 || realBox.bottom-realBox.top < 5) return;
          recompBtn.disabled = true; recompBtn.textContent = "Re-composing…";
          addTyping();

          fetch("/api/recompose", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
              source_image: _sourcePayload,
              preview_image: _sourcePayload,
              bbox: realBox,
              target: stepData.step.target,
              verb: stepData.step.verb,
              custom_instruction: (document.getElementById("instr-" + editorId) || {}).value || "",
            })
          })
          .then(function(r) { return r.json(); })
          .then(function(data) {
            removeTyping();
            if (data.error) { addErrorMsg(data.error); recompBtn.disabled = false; recompBtn.textContent = "Re-compose"; return; }
            var q = data.quality;
            var imgs = addImageComparison(data.overlay_image, data.final_image, "Region", "Result");
            var metrics = addMetrics(q);

            addToolMsg("blend_back", "Re-composing with adjusted bounding box",
              "BBox: " + data.bbox.left + "," + data.bbox.top + " → " + data.bbox.right + "," + data.bbox.bottom);
            addAgentMsg("🔄 Re-composed result:", imgs + metrics);

            _sourcePayload = data.final_image;
            addToSidebar(data.final_image);
            _iterCount++;
            createBboxEditor(data.final_image, data.bbox, imgW, imgH, stepIdx + "_i" + _iterCount, stepData);
            recompBtn.disabled = false; recompBtn.textContent = "Re-compose";
          })
          .catch(function(err) {
            removeTyping();
            addErrorMsg(err.message);
            recompBtn.disabled = false; recompBtn.textContent = "Re-compose";
          });
        });

        drawBox();
      }

      if (img.complete) setup(); else img.onload = setup;
    }, 100);
  }
</script>
</body>
</html>
"""


def make_handler(app: AgentBananaApp) -> Callable[..., BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self._send_html(HTML_PAGE)
                return
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON payload"})
                return

            if self.path == "/api/edit-stream":
                # SSE streaming endpoint — sends each agent step as it completes
                instruction = str(payload.get("instruction", "")).strip()
                image_payload = str(payload.get("image", "")).strip()
                session_id = payload.get("session_id") or None
                if not instruction:
                    self._send_json(400, {"error": "Instruction is required"})
                    return
                if not image_payload:
                    self._send_json(400, {"error": "Image payload is required"})
                    return

                # Set up SSE response headers
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                def send_step(step):
                    """Emit a single agent step as an SSE event."""
                    try:
                        step_data = step.to_dict()
                        event_str = f"data: {json.dumps(step_data)}\n\n"
                        self.wfile.write(event_str.encode("utf-8"))
                        self.wfile.flush()
                    except Exception:
                        pass  # Client may have disconnected

                try:
                    image = decode_image_payload(image_payload)
                    result = app.run(image, instruction, session_id=session_id,
                                     step_callback=send_step)
                    # Send final result as a 'done' event
                    done_str = f"event: done\ndata: {json.dumps(result.to_dict())}\n\n"
                    self.wfile.write(done_str.encode("utf-8"))
                    self.wfile.flush()
                except Exception as exc:
                    err_str = f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
                    self.wfile.write(err_str.encode("utf-8"))
                    self.wfile.flush()

            elif self.path == "/api/ground":
                # Manual mode step 1: parse instruction + ground targets
                instruction = str(payload.get("instruction", "")).strip()
                image_payload = str(payload.get("image", "")).strip()
                if not instruction or not image_payload:
                    self._send_json(400, {"error": "instruction and image are required"})
                    return
                try:
                    image = decode_image_payload(image_payload)
                    result = app.ground_targets(image, instruction)
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                    return
                self._send_json(200, result)

            elif self.path == "/api/edit-manual-stream":
                # Manual mode step 2: run edits with user-confirmed bboxes (SSE)
                instruction = str(payload.get("instruction", "")).strip()
                image_payload = str(payload.get("image", "")).strip()
                confirmed_bboxes = payload.get("bboxes", [])
                session_id = payload.get("session_id") or None
                if not instruction or not image_payload:
                    self._send_json(400, {"error": "instruction and image are required"})
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                def send_step_manual(step):
                    try:
                        event_str = f"data: {json.dumps(step.to_dict())}\n\n"
                        self.wfile.write(event_str.encode("utf-8"))
                        self.wfile.flush()
                    except Exception:
                        pass

                try:
                    image = decode_image_payload(image_payload)
                    result = app.run_with_bboxes(
                        image, instruction, confirmed_bboxes,
                        session_id=session_id, step_callback=send_step_manual,
                    )
                    done_str = f"event: done\ndata: {json.dumps(result.to_dict())}\n\n"
                    self.wfile.write(done_str.encode("utf-8"))
                    self.wfile.flush()
                except Exception as exc:
                    err_str = f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
                    self.wfile.write(err_str.encode("utf-8"))
                    self.wfile.flush()

            elif self.path == "/api/edit":
                instruction = str(payload.get("instruction", "")).strip()
                image_payload = str(payload.get("image", "")).strip()
                session_id = payload.get("session_id") or None
                if not instruction:
                    self._send_json(400, {"error": "Instruction is required"})
                    return
                if not image_payload:
                    self._send_json(400, {"error": "Image payload is required"})
                    return
                try:
                    image = decode_image_payload(image_payload)
                    result = app.run(image, instruction, session_id=session_id)
                except Exception as exc:  # pragma: no cover
                    self._send_json(500, {"error": str(exc)})
                    return
                self._send_json(200, result.to_dict())

            elif self.path == "/api/recompose":
                source_payload = str(payload.get("source_image", "")).strip()
                preview_payload = str(payload.get("preview_image", "")).strip()
                print(f"[agent-banana] recompose: source_len={len(source_payload)}, preview_len={len(preview_payload)}, bbox={payload.get('bbox')}")
                bbox_data = payload.get("bbox")
                target = str(payload.get("target", "object"))
                verb = str(payload.get("verb", "edit"))
                if not source_payload or not bbox_data:
                    self._send_json(400, {"error": "source_image and bbox are required"})
                    return
                if not preview_payload:
                    preview_payload = source_payload
                try:
                    source_image = decode_image_payload(source_payload)
                    preview_image = decode_image_payload(preview_payload)
                    custom_instruction = str(payload.get("custom_instruction", "")).strip()
                    result = app.recompose(source_image, preview_image, bbox_data, target, verb, custom_instruction=custom_instruction)
                except Exception as exc:  # pragma: no cover
                    self._send_json(500, {"error": str(exc)})
                    return
                self._send_json(200, result)

            else:
                self._send_json(404, {"error": "Not found"})

        def log_message(self, format: str, *args: object) -> None:
            print(f"[agent-banana] {self.address_string()} - {format % args}")

    return Handler


def main() -> None:
    default_root = Path(os.environ.get("AGENT_ROOT", "/tmp/agent_banana"))
    parser = argparse.ArgumentParser(description="Serve the Agent Banana image editing demo over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    app = AgentBananaApp.from_env(default_root)
    server = ThreadedHTTPServer((args.host, args.port), make_handler(app))
    print(f"Serving Agent Banana Studio on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

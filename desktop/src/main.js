const promptInput = document.getElementById("promptInput");
const promptIcon = document.getElementById("promptIcon");
const spinner = document.getElementById("spinner");
const micBtn = document.getElementById("micBtn");
const closeBtn = document.getElementById("closeBtn");
const responsePanel = document.getElementById("responsePanel");
const responseContent = document.getElementById("responseContent");
const footerMeta = document.getElementById("footerMeta");

const COMPACT_HEIGHT = 68;
const EXPANDED_HEIGHT = 420;

let isRecording = false;

function setLoading(loading) {
  if (loading) {
    promptIcon.classList.add("hidden");
    spinner.classList.remove("hidden");
  } else {
    spinner.classList.add("hidden");
    promptIcon.classList.remove("hidden");
  }
}

async function invokeTauri(command, args = {}) {
  if (window.__TAURI__?.core?.invoke) {
    return await window.__TAURI__.core.invoke(command, args);
  }
  if (window.__TAURI_INTERNALS__?.invoke) {
    return await window.__TAURI_INTERNALS__.invoke(command, args);
  }
  if (window.__TAURI__?.invoke) {
    return await window.__TAURI__.invoke(command, args);
  }

  if (command === "send_query") {
    const res = await fetch("http://127.0.0.1:8765/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: args.prompt }),
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    }
    return await res.json();
  }

  if (command === "record_voice") {
    const res = await fetch("http://127.0.0.1:8765/api/voice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record_seconds: args.seconds || 4.0 }),
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    }
    return await res.json();
  }

  if (command === "hide_window") {
    try {
      await fetch("http://127.0.0.1:8766/hide", { method: "POST", mode: "no-cors" });
    } catch (_) {
      // Ignore if IPC listener is not active
    }
    return;
  }

  if (command === "set_window_height") {
    return;
  }

  throw new Error(`Unsupported command: ${command}`);
}

async function setWindowHeight(height) {
  try {
    await invokeTauri("set_window_height", { height });
  } catch (_) {
    // Ignore when running outside Tauri
  }
}

async function showResultsPanel() {
  responsePanel.classList.remove("hidden");
  await setWindowHeight(EXPANDED_HEIGHT);
}

async function hideResultsPanel() {
  responsePanel.classList.add("hidden");
  await setWindowHeight(COMPACT_HEIGHT);
}

async function closeWindow() {
  await hideResultsPanel();
  promptInput.value = "";
  await invokeTauri("hide_window").catch(() => {});
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatMessage(text) {
  if (!text) return "";

  if (text.includes("429") || text.includes("OTPM") || text.includes("Request too large")) {
    return `
      <div class="error-notice">
        Límite de tokens alcanzado en el proveedor actual. Probá con una consulta más breve o cambiá el modelo/proveedor en <code>.env</code>.
      </div>
    `;
  }

  let html = escapeHtml(text);

  html = html.replace(/```([\w]*)\n([\s\S]*?)```/g, (_match, _lang, code) => {
    return `<pre><code>${code.trim()}</code></pre>`;
  });
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/^\s*[-*]\s+(.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>[\s\S]*?<\/li>)/g, "<ul>$1</ul>");
  html = html.replace(/\n\n+/g, "<br><br>");
  html = html.replace(/\n/g, "<br>");

  return html;
}

function formatError(msg) {
  return `<div class="error-notice">${escapeHtml(msg)}</div>`;
}

function updateFooter(intent) {
  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (intent && intent.label && intent.label !== "unknown") {
    const conf = Math.round((intent.confidence || 0) * 100);
    footerMeta.textContent = `${intent.label} · ${conf}% · ${time}`;
  } else {
    footerMeta.textContent = time;
  }
}

async function streamQuery(prompt, onToken) {
  const res = await fetch("http://127.0.0.1:8765/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";

    for (const block of blocks) {
      let eventType = "message";
      let dataLine = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        if (line.startsWith("data:")) dataLine += line.slice(5).trim();
      }
      if (!dataLine) continue;
      const payload = JSON.parse(dataLine);
      if (eventType === "token" && payload.token) {
        onToken(payload.token);
      } else if (eventType === "done") {
        finalResult = payload;
      } else if (eventType === "error") {
        return { error: payload.error };
      }
    }
  }

  return finalResult || { response: "" };
}

async function handleSubmit() {
  const query = promptInput.value.trim();
  if (!query) return;

  setLoading(true);
  await showResultsPanel();
  responseContent.textContent = "Procesando...";
  footerMeta.textContent = "vaquita";

  let accumulated = "";

  try {
    let result;
    try {
      result = await streamQuery(query, (token) => {
        accumulated += token;
        responseContent.textContent = accumulated;
        responsePanel.scrollTop = responsePanel.scrollHeight;
      });
    } catch (_) {
      result = await invokeTauri("send_query", { prompt: query });
    }

    setLoading(false);
    updateFooter(result.intent);

    if (result.error) {
      responseContent.innerHTML = formatError(result.error);
    } else {
      responseContent.innerHTML = formatMessage(result.response || accumulated);
    }
  } catch (err) {
    setLoading(false);
    footerMeta.textContent = "sin conexión";
    responseContent.innerHTML = formatError(String(err.message || err));
  }
}

async function handleVoiceRecord() {
  if (isRecording) return;
  isRecording = true;
  micBtn.classList.add("recording");
  setLoading(true);
  promptInput.placeholder = "Escuchando...";
  await showResultsPanel();
  responseContent.textContent = "Grabando comando de voz...";
  footerMeta.textContent = "micrófono";

  try {
    const result = await invokeTauri("record_voice", { seconds: 4.0 });
    setLoading(false);
    micBtn.classList.remove("recording");
    promptInput.placeholder = "Ask Vaquita...";

    if (result.transcribed_prompt) {
      promptInput.value = result.transcribed_prompt;
    }

    updateFooter(result.intent);

    if (result.error) {
      responseContent.innerHTML = formatError(result.error);
    } else {
      responseContent.innerHTML = formatMessage(result.response || "");
    }
  } catch (err) {
    setLoading(false);
    micBtn.classList.remove("recording");
    promptInput.placeholder = "Ask Vaquita...";
    footerMeta.textContent = "error";
    responseContent.innerHTML = formatError(String(err.message || err));
  } finally {
    isRecording = false;
  }
}

promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    handleSubmit();
  }
});

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    e.preventDefault();
    if (!responsePanel.classList.contains("hidden")) {
      hideResultsPanel();
    } else {
      closeWindow();
    }
  } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "m") {
    e.preventDefault();
    handleVoiceRecord();
  }
});

micBtn.addEventListener("click", handleVoiceRecord);
closeBtn.addEventListener("click", closeWindow);

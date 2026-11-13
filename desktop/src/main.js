const promptInput = document.getElementById("promptInput");
const sendBtn = document.getElementById("sendBtn");
const statusOrb = document.getElementById("statusOrb");
const responsePanel = document.getElementById("responsePanel");
const responseContent = document.getElementById("responseContent");
const intentPill = document.getElementById("intentPill");
const intentText = document.getElementById("intentText");
const hudTimestamp = document.getElementById("hudTimestamp");

function setStatus(state) {
  statusOrb.className = "status-orb " + state;
}

async function invokeTauri(command, args) {
  if (window.__TAURI__ && window.__TAURI__.core && window.__TAURI__.core.invoke) {
    return await window.__TAURI__.core.invoke(command, args);
  }
  if (window.__TAURI__ && window.__TAURI__.invoke) {
    return await window.__TAURI__.invoke(command, args);
  }

  if (command === "send_query") {
    const res = await fetch("http://127.0.0.1:8765/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: args.prompt }),
    });
    if (!res.ok) {
      throw new Error(`Daemon error: HTTP ${res.status}`);
    }
    return await res.json();
  }

  if (command === "hide_window") {
    responsePanel.classList.add("hidden");
    return;
  }

  throw new Error(`Unsupported command: ${command}`);
}

async function handleSubmit() {
  const query = promptInput.value.trim();
  if (!query) return;

  setStatus("thinking");
  responsePanel.classList.remove("hidden");
  responseContent.textContent = "Analizando contexto...";
  intentText.textContent = "neural // routing...";
  hudTimestamp.textContent = new Date().toLocaleTimeString();

  try {
    const result = await invokeTauri("send_query", { prompt: query });
    setStatus("success");

    if (result.intent && result.intent.label) {
      const conf = Math.round((result.intent.confidence || 0) * 100);
      intentText.textContent = `${result.intent.label} • ${conf}%`;
      intentPill.style.display = "inline-flex";
    } else {
      intentPill.style.display = "none";
    }

    if (result.error) {
      responseContent.textContent = "Error: " + result.error;
      setStatus("error");
    } else {
      responseContent.textContent = result.response || "Completado.";
    }
  } catch (err) {
    setStatus("error");
    intentText.textContent = "error // offline";
    responseContent.textContent = String(err.message || err);
  }
}

// Event Listeners
promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    handleSubmit();
  } else if (e.key === "Escape") {
    e.preventDefault();
    if (!responsePanel.classList.contains("hidden")) {
      responsePanel.classList.add("hidden");
    } else {
      promptInput.value = "";
      invokeTauri("hide_window", {}).catch(() => {});
    }
  }
});

sendBtn.addEventListener("click", handleSubmit);

/* Dev and Tasks describe reuse Chat's composer globals + listeners (attach, paste, mic,
 * send). Load AFTER state.js + api.js + attachments.js + voice.js — Chat's
 * modules, not a second client. Mic is Chat's toggleRecording (voice.js).
 * Send uses the same send-btn click as Chat; the handler starts the run so
 * Kind still applies. Do not redeclare `$` / `state` / `BASE`. */
const messageInput = $("#message-input");
const micBtn = $("#mic-btn");
const sendBtn = $("#send-btn");
const pastePreview = $("#paste-preview");
const realtimeBtn = null;
const realtimeListeningCaption = null;
const describeHost = document.querySelector("[data-describe-form]") || document.getElementById("sdlc-composer");
const describePlaceholder = messageInput ? messageInput.placeholder : "";

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function escapeAttr(s) { return escapeHtml(s); }
function appendSystemMessage(text) {
  const status = document.getElementById("status-text");
  if (!status) return;
  status.textContent = String(text || "");
}
function updateSendButton() {
  if (!sendBtn) return;
  const pending = typeof askUserHasPendingCard === "function"
    && askUserHasPendingCard();
  const draft = typeof composerHasDraft === "function" && composerHasDraft();
  const stopping = !!(pending && !draft);
  const sendIcon = sendBtn.querySelector(".send-icon");
  const stopIcon = sendBtn.querySelector(".stop-icon");
  sendBtn.classList.toggle("stop-btn", stopping);
  sendBtn.classList.toggle("steer-btn", !!(pending && draft));
  if (stopping) {
    sendBtn.disabled = false;
    sendBtn.title = "Abort pending question";
    sendBtn.setAttribute("aria-label", sendBtn.title);
    if (sendIcon) sendIcon.style.display = "none";
    if (stopIcon) stopIcon.style.display = "";
  } else {
    sendBtn.title = pending && draft
      ? "Steer — abort the pending question and send this message"
      : "Send message";
    sendBtn.setAttribute("aria-label", sendBtn.title);
    if (stopIcon) stopIcon.style.display = "none";
    if (sendIcon) sendIcon.style.display = "";
  }
  const tree = document.getElementById("tree-stop-btn");
  if (tree) {
    tree.hidden = !(pending && draft);
    tree.disabled = false;
    tree.title = "Abort pending question";
    tree.setAttribute("aria-label", tree.title);
  }
}
function syncTreeStopBtn() { updateSendButton(); }

const TEXT_EXT = {
  ".txt": 1, ".md": 1, ".csv": 1, ".tsv": 1, ".json": 1, ".log": 1,
  ".yaml": 1, ".yml": 1, ".py": 1, ".js": 1, ".ts": 1, ".sh": 1,
  ".sql": 1, ".html": 1, ".css": 1, ".diff": 1, ".patch": 1,
};

function utf8ToB64(text) {
  return btoa(unescape(encodeURIComponent(String(text || ""))));
}
function dataUrlToB64(dataUrl) {
  const s = String(dataUrl || "");
  const i = s.indexOf(",");
  return i >= 0 ? s.slice(i + 1) : s;
}
function textName(name) {
  const n = String(name || "attachment.md");
  const ext = (n.match(/\.[a-z0-9]+$/i) || [""])[0].toLowerCase();
  if (TEXT_EXT[ext]) return n;
  return (n.replace(/\.[^.]+$/, "") || "attachment") + ".md";
}

function waitPending() {
  const pending = []
    .concat(state.pendingImageReads || [])
    .concat(state.pendingTextReads || []);
  return pending.length ? Promise.all(pending) : Promise.resolve();
}

function composerPayload() {
  const out = [];
  (state.pastedImages || []).forEach((img) => {
    const data = dataUrlToB64(img.dataUrl || img.url || "");
    if (!data) return;
    out.push({ name: img.name || "pasted.png", data_base64: data });
  });
  (state.attachedTextFiles || []).forEach((f) => {
    out.push({ name: textName(f.name), data_base64: utf8ToB64(f.content || "") });
  });
  return out;
}

function setFrozen(frozen) {
  if (messageInput) {
    messageInput.readOnly = !!frozen;
    messageInput.placeholder = frozen
      ? ""
      : describePlaceholder;
  }
  ["attach-btn", "attach-input", "mic-btn", "send-btn"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = !!frozen;
  });
  if (typeof updateSendState === "function") updateSendState();
}

function clearComposer() {
  if (typeof clearAttachments === "function") clearAttachments();
  if (messageInput) messageInput.value = "";
  const status = document.getElementById("status-text");
  if (status) status.textContent = "";
  if (typeof updateSendState === "function") updateSendState();
}

/* Chat send clicks sendMessage() directly. A native form submit with no
 * preventDefault GETs /dev? and wipes the page. Call the Dev start handler
 * the same way Chat calls sendMessage. */
function submitDescribe() {
  if (typeof window.describeSubmit === "function") return window.describeSubmit();
  if (messageInput && messageInput.readOnly) return;
  if (typeof window.devStartDescribe === "function") window.devStartDescribe();
}

window.devComposer = {
  text: function () { return (messageInput && messageInput.value) || ""; },
  setText: function (value) {
    if (messageInput) messageInput.value = String(value || "");
    if (typeof updateSendState === "function") updateSendState();
  },
  payload: composerPayload,
  waitPending: waitPending,
  setFrozen: setFrozen,
  clear: clearComposer,
  hasDraft: function () {
    return typeof composerHasDraft === "function"
      ? composerHasDraft()
      : !!(messageInput && messageInput.value.trim());
  },
  submit: submitDescribe,
};
window.describeComposer = window.devComposer;

function bindDevComposer() {
  const form = document.getElementById(describeHost?.dataset.describeForm || "sdlc-start");
  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      if (describeHost?.dataset.describeForm) submitDescribe();
    }, true);
  }

  const attachBtn = $("#attach-btn");
  const attachInput = $("#attach-input");
  if (attachBtn && attachInput) {
    attachBtn.addEventListener("click", () => attachInput.click());
    attachInput.addEventListener("change", () => {
      if (typeof attachFiles === "function") {
        attachFiles(Array.from(attachInput.files || []));
      }
      attachInput.value = "";
    });
  }

  if (messageInput) {
    messageInput.addEventListener("input", () => {
      if (typeof updateSendState === "function") updateSendState();
    });
    messageInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        submitDescribe();
      }
    });
    messageInput.addEventListener("paste", (e) => {
      if (messageInput.readOnly) return;
      const items = e.clipboardData && e.clipboardData.items;
      if (!items) return;
      const files = [];
      for (const item of items) {
        if (item.kind !== "file") continue;
        const f = item.getAsFile();
        if (f) files.push(f);
      }
      if (!files.length) return;
      e.preventDefault();
      if (typeof attachFiles === "function") attachFiles(files);
    });
  }

  if (sendBtn) {
    sendBtn.addEventListener("click", (e) => {
      e.preventDefault();
      const pending = typeof askUserHasPendingCard === "function"
        && askUserHasPendingCard();
      const draft = typeof composerHasDraft === "function" && composerHasDraft();
      if (pending && !draft) {
        if (typeof abortAllAskUserCards === "function") abortAllAskUserCards();
        return;
      }
      submitDescribe();
    });
  }
  const treeStop = document.getElementById("tree-stop-btn");
  if (treeStop) {
    treeStop.addEventListener("click", (e) => {
      e.preventDefault();
      if (typeof abortAllAskUserCards === "function") abortAllAskUserCards();
    });
  }

  if (pastePreview) {
    pastePreview.addEventListener("click", (e) => {
      if (messageInput && messageInput.readOnly) return;
      let btn = e.target.closest(".paste-thumb-remove");
      if (btn) {
        const idx = parseInt(btn.dataset.idx, 10);
        if (!isNaN(idx) && idx >= 0 && idx < state.pastedImages.length) {
          state.pastedImages.splice(idx, 1);
          if (typeof renderPastePreview === "function") renderPastePreview();
        }
        return;
      }
      btn = e.target.closest(".file-remove");
      if (btn) {
        const idx = parseInt(btn.dataset.idx, 10);
        if (!isNaN(idx) && idx >= 0 && idx < state.attachedTextFiles.length) {
          state.attachedTextFiles.splice(idx, 1);
          if (typeof renderPastePreview === "function") renderPastePreview();
        }
      }
    });
  }

  const host = describeHost;
  if (host) {
    const stop = (ev) => { ev.preventDefault(); ev.stopPropagation(); };
    ["dragenter", "dragover"].forEach((name) => {
      host.addEventListener(name, (ev) => {
        stop(ev);
        host.classList.add("dropping");
      });
    });
    ["dragleave", "drop"].forEach((name) => {
      host.addEventListener(name, (ev) => {
        stop(ev);
        if (name === "dragleave" && host.contains(ev.relatedTarget)) return;
        host.classList.remove("dropping");
      });
    });
    host.addEventListener("drop", (ev) => {
      if (messageInput && messageInput.readOnly) return;
      const files = (ev.dataTransfer || {}).files;
      if (files && files.length && typeof attachFiles === "function") {
        attachFiles(Array.from(files));
      }
    });
  }

  if (micBtn && typeof toggleRecording === "function") {
    micBtn.addEventListener("click", toggleRecording);
  }
  if (typeof initMic === "function") initMic();
  if (typeof updateSendState === "function") updateSendState();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bindDevComposer);
} else {
  bindDevComposer();
}

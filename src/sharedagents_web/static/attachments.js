/* attachments.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */
function composerHasDraft() {
  const hasText = !!(messageInput && messageInput.value.trim());
  return hasText || state.pastedImages.length > 0 || state.attachedTextFiles.length > 0;
}

/** Parent turn streaming OR any live descendant in the session tree. */
function treeBusy() {
  return !!(state.isStreaming || state.treeLive);
}

/** True when a live turn should be cancelled and the draft (with images)
 *  submitted as a fresh prompt. The durable queue is text-only, so vision
 *  blocks cannot ride `/queue` — Steer is the working path. */
function composerWillSteerBusy() {
  if (!state.currentSessionId || state.monitorReadOnly) return false;
  if (!(state.pastedImages && state.pastedImages.length)) return false;
  return !!(state.isStreaming || state.treeLive);
}

/** True when clicking the primary button RIGHT NOW would enqueue a follow-up
 *  rather than stop the turn. This must mirror the streaming branch of
 *  sendMessage() EXACTLY, or the button lies (shows Queue but a click aborts).
 *  Queue only while the PARENT turn is still open (isStreaming). After the
 *  parent ends but subagents remain (treeLive), typed text Sends a new jarvis
 *  message — same as idle — while tree-stop-btn kills the tree.
 *  Images steer instead (see composerWillSteerBusy). Slash commands do not
 *  queue. Saved files ride the queued follow-up as references. */
function composerWillQueue() {
  // After Stop on a parked question the old turn can still look streaming.
  // Queueing the follow-up waits for that turn forever — no assistant reply.
  if (state._askStopReleased) return false;
  if (!state.isStreaming || !state.currentSessionId) return false;
  if (state.pastedImages && state.pastedImages.length) return false;
  const t = messageInput ? messageInput.value.trim() : "";
  if (t.startsWith("/")) return false;
  return !!t || !!(state.attachedTextFiles && state.attachedTextFiles.length);
}

/** Single source of truth for the send button's enabled state: text OR a
 *  staged image makes the prompt sendable. Every writer of sendBtn.disabled
 *  outside the streaming toggle must go through this.
 *
 *  While the session tree is busy the button is always actionable — Stop
 *  (empty composer) or Queue-follow-up (has a draft) — so it must never be
 *  disabled for lack of text; otherwise clearing the box would grey out Stop. */
function updateSendState() {
  if (!sendBtn) return;
  sendBtn.disabled =
    !!state.monitorReadOnly || !!state.queueSubmitting ||
    // The first Send click is still draining the armed mic into the composer.
    !!state.voiceSendFlushing ||
    // Hold Send while the first click is still creating a session / applying
    // the model. updateSendButton re-enables once isStreaming so Stop works.
    (!!state.promptSubmitting && !state.isStreaming) ||
    !!(state.pendingPrompt && state.pendingPrompt.sid === state.currentSessionId) ||
    (!composerHasDraft() && !state.currentSessionId);
  // Keep the icon/label in lock-step with the draft so typing while a turn /
  // subagent tree runs flips the red Stop into a Send/queue affordance.
  updateSendButton();
  if (typeof syncTreeStopBtn === "function") syncTreeStopBtn();
}

/** Type-keyed emoji for an attached-file pill, by extension. */
function fileIcon(name) {
  const ext = (String(name || "").match(/\.([a-z0-9]+)$/i) || [, ""])[1].toLowerCase();
  if (ext === "pdf") return "📕";
  if (ext === "doc" || ext === "docx" || ext === "odt" || ext === "rtf") return "📝";
  if (ext === "xls" || ext === "xlsx" || ext === "ods" || ext === "csv" || ext === "tsv") return "📊";
  if (ext === "ppt" || ext === "pptx" || ext === "odp") return "📑";
  if (ext === "md" || ext === "txt" || ext === "json" || ext === "xml" ||
      ext === "html" || ext === "htm" || ext === "epub") return "📄";
  return "📎";
}

/** Saved uploads travel as file references beside the typed text. The
 *  backend builds the model-facing attachment context from them; the page
 *  only forwards what the upload route returned. */
function _attachmentRefs(files, images = []) {
  return [...(files || []), ...(images || [])]
    .filter((file) => file && file.path)
    .map((file) => ({
      path: file.path,
      name: file.name || "",
      mime: file.mime || "",
      size: Number(file.size) || 0,
    }));
}

/** Chips for the optimistic You-bubble: one per saved non-image file. */
function _attachmentChips(files) {
  return (files || []).map((file) => ({
    name: file.name || "attachment",
    size: Number(file.size) || 0,
    mime: file.mime || "",
  }));
}

/** Match a canonical turn_started prompt to an optimistic local echo. */
function _pendingEchoMatches(pending, sid, canonicalText, imageCount) {
  if (!pending) return false;
  const pendingImages = Number(pending.images || 0);
  const canonicalImages = Number(imageCount || 0);
  // turn_started often omits image blocks from ``user``. That is still the
  // local send — a canonical echo with MORE images is a different source.
  if (canonicalImages > pendingImages) return false;
  const sidOk = !pending.sid || !sid || pending.sid === sid;
  const norm = (value) => String(value || "")
    .replace(/\s+/g, " ").trim().toLowerCase();
  const stripAttach = (value) => norm(value).replace(
    /^(?:[^\n]*attached:\s*[^\n]+\n*)+/i, "",
  );
  const pendingText = String(pending.text || "");
  const pendingEcho = String(pending.echoText || pending.text || "");
  const canonical = String(canonicalText || "");
  const compactPair = (
    typeof _isCompactUserText === "function" &&
    _isCompactUserText(pendingText) &&
    _isCompactUserText(canonical)
  );
  const nPending = norm(pendingText);
  const nEcho = norm(pendingEcho);
  const nCanon = norm(canonical);
  const textOk = compactPair
    || nPending === nCanon
    || nEcho === nCanon
    || !!(nCanon && nPending.endsWith(nCanon))
    || !!(stripAttach(pendingEcho) && stripAttach(pendingEcho) === stripAttach(canonical));
  if (sidOk && textOk) return true;
  // Same-tab send: the optimistic You-bubble is still the last user row.
  // Session id can change on first-intent materialize, and image-only
  // turn_started text can disagree — still the click we just painted.
  const last = (typeof state !== "undefined" && state && Array.isArray(state.msgCache))
    ? state.msgCache[state.msgCache.length - 1]
    : null;
  return !!(pending.el && last && last.role === "user" && last.el === pending.el);
}

/** Render thumbnail previews for staged images + text-file cards (remove is
 *  delegated — one listener bound at init, not re-bound per render). */
function renderPastePreview() {
  if (!pastePreview) return;
  const imgs = state.pastedImages;
  const files = state.attachedTextFiles;
  const hasAny = imgs.length || files.length;
  pastePreview.hidden = !hasAny;

  let html = imgs
    .map(
      (img, i) => {
        const src = escapeHtml(img.url || img.dataUrl || "");
        const name = escapeHtml(img.name || "pasted image");
        // data-artifact-url prefers durable /files URL when already uploaded.
        const openUrl = escapeHtml(img.url || img.dataUrl || "");
        return `
    <div class="paste-thumb" data-idx="${i}" data-artifact-url="${openUrl}" data-artifact-name="${name}">
      <img class="paste-thumb-img" src="${src}" alt="${name}">
      <button class="paste-thumb-remove" data-idx="${i}" type="button" title="Remove image">&times;</button>
    </div>
  `;
      },
    )
    .join("");

  // Render attached files as compact pills (icon · name · size). The converted
  // contents live in f.content and are sent to the model at submit — they are
  // NOT shown in the chip (a converted PDF/CSV would otherwise dump a wall of
  // markdown into the input).
  if (files.length) {
    html += files.map((f, i) => {
      const sizeStr = f.size < 1024
        ? `${f.size} B`
        : `${(f.size / 1024).toFixed(1)} KB`;
      const meta = sizeStr + (f.path ? " · saved" : "") + (f.truncated ? " · truncated" : "");
      return `
    <div class="paste-file-thumb" data-idx="${i}" title="${escapeAttr(f.name)}">
      <span class="file-icon" aria-hidden="true">${fileIcon(f.name)}</span>
      <span class="file-info">
        <span class="file-name">${escapeHtml(f.name)}</span>
        <span class="file-size">${escapeHtml(meta)}</span>
      </span>
      <button class="file-remove" data-idx="${i}" type="button" title="Remove file">&times;</button>
    </div>
  `;
    }).join("");
  }

  pastePreview.innerHTML = html;
  updateSendState();
}

/** Clear all staged attachments (images + text files) from state and preview. */
function clearAttachments() {
  state.pastedImages = [];
  state.attachedTextFiles = [];
  state.pendingTextReads = [];
  if (pastePreview) {
    pastePreview.hidden = true;
    pastePreview.innerHTML = "";
  }
  updateSendState();
}

/** Legacy alias — clearAttachments() is the canonical name now. */
function clearPastedImages() { clearAttachments(); }

// Base64 data-URLs inflate ~1.37× over raw and ride a JSON body capped at
// 10MB server-side (and become one jsonl line in the session log) — budget
// well under the cap and downscale screenshots instead of failing late.
const ATTACH_TOTAL_BUDGET = 8 * 1024 * 1024;
const ATTACH_MAX_DIM = 1600;

function _readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (ev) => resolve(ev.target.result);
    reader.onerror = () => reject(new Error("could not read file"));
    reader.readAsDataURL(file);
  });
}

/** Read an image file; large ones are downscaled to ≤1600px JPEG so a 12MP
 *  screenshot doesn't blow the body cap or bloat the session log. */
async function _normalizeImage(file, mime) {
  const original = await _readFileAsDataUrl(file);
  const dataUrl = mime
    ? original.replace(/^data:[^;,]*;base64,/, `data:${mime};base64,`)
    : original;
  if (file.size <= 512 * 1024) return dataUrl;
  const img = new Image();
  await new Promise((res, rej) => {
    img.onload = res;
    img.onerror = rej;
    img.src = dataUrl;
  });
  const scale = Math.min(
    1,
    ATTACH_MAX_DIM / Math.max(img.width, img.height, 1),
  );
  if (scale === 1 && file.size <= 2 * 1024 * 1024) return dataUrl;
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(img.width * scale));
  canvas.height = Math.max(1, Math.round(img.height * scale));
  canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.85);
}

/** Save originals for every file type. Documents remain on disk for the agent
 *  to sample; image previews are optional and never replace the saved file.
 *  API clients can also supply destination_dir directly to /api/attachments. */
function attachFiles(files, destinationDir) {
  const destination = String(destinationDir || "").trim();
  for (const file of files) {
    if (!file) continue;
    if (file.size > 25 * 1024 * 1024) {
      appendSystemMessage(
        `"${escapeHtml(file.name || "file")}" not attached — files must be under 25 MB.`,
        true,
      );
      continue;
    }
    const read = (async () => {
      try {
        const fd = new FormData();
        fd.append("file", file, file.name || "attachment");
        fd.append("extract_text", "false");
        if (destination) fd.append("destination_dir", destination);
        const resp = await fetch(BASE + "/api/attachments", {
          method: "POST", body: fd, credentials: "same-origin",
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok || !data.path) {
          const detail = data.error || (resp.status === 413
            ? "file exceeds the server or proxy upload limit (HTTP 413)"
            : `invalid upload response (HTTP ${resp.status})`);
          appendSystemMessage(
            `Could not save "${escapeHtml(file.name || "file")}" — ${escapeHtml(detail)}.`,
            true,
          );
          return;
        }
        const saved = {
          content: "", path: data.path,
          name: data.name || file.name || "attachment",
          size: Number(data.size) || file.size || 0,
          mime: data.mime || "",
        };
        // Use byte-sniffed raster types from the server rather than trusting
        // extensions or MIME declarations. Preserve the full original even
        // when the inline preview is downscaled or exceeds the vision budget.
        if (["image/png", "image/jpeg", "image/gif", "image/webp"].includes(data.image_mime)) {
          try {
            const dataUrl = await _normalizeImage(file, data.image_mime);
            const total = state.pastedImages.reduce((n, img) => n + img.dataUrl.length, 0);
            if (total + dataUrl.length <= ATTACH_TOTAL_BUDGET) {
              state.pastedImages.push({
                dataUrl, name: saved.name, path: saved.path,
                size: saved.size, mime: data.image_mime,
              });
              renderPastePreview();
              return;
            }
          } catch (_) { /* saved original still attaches as a file below */ }
        }
        state.attachedTextFiles.push(saved);
        renderPastePreview();
      } catch (_) {
        appendSystemMessage(
          `Could not upload "${escapeHtml(file.name || "file")}".`, true,
        );
      }
    })();
    state.pendingTextReads.push(read);
  }
}

function initMic() {
  if (!micBtn) return;
  const hasMic = !!(
    navigator.mediaDevices && navigator.mediaDevices.getUserMedia
  );
  if (!hasMic) {
    state.audioState = "unsupported";
    micBtn.disabled = true;
    micBtn.title = "Microphone not available (HTTPS required)";
  } else {
    micBtn.disabled = state.isStreaming; // disabled while a turn runs
  }
}

// Poll gateway reachability and reflect it in the header badge.

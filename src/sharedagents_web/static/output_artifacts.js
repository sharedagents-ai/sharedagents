/* Agent-produced image/file outputs.

   The backend projects only display metadata + SHA-256. Local paths remain on
   the server; every URL below is inside the authenticated /files asset jail.
   */

const OUTPUT_IMAGE_TYPES = new Set([
  "image/png", "image/jpeg", "image/gif", "image/webp",
]);

const OUTPUT_TYPE_LABELS = new Map([
  ["image/png", "PNG"],
  ["image/jpeg", "JPEG"],
  ["image/gif", "GIF"],
  ["image/webp", "WebP"],
  ["application/pdf", "PDF"],
  ["text/plain", "Text"],
  ["text/markdown", "Markdown"],
  ["text/csv", "CSV"],
  ["text/tab-separated-values", "TSV"],
  ["application/json", "JSON"],
  ["application/msword", "Word"],
  ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "Word"],
  ["application/vnd.ms-excel", "Excel"],
  ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "Excel"],
  ["application/vnd.ms-powerpoint", "PowerPoint"],
  ["application/vnd.openxmlformats-officedocument.presentationml.presentation", "PowerPoint"],
  ["application/zip", "ZIP"],
  ["application/x-7z-compressed", "7Z"],
  ["application/x-tar", "TAR"],
  ["application/gzip", "GZIP"],
]);

function _artifactSizeLabel(value) {
  const size = Number(value || 0);
  if (!Number.isFinite(size) || size <= 0) return "";
  if (size < 1024) return `${Math.round(size)} B`;
  if (size < 1024 * 1024) {
    const kb = size / 1024;
    return `${kb >= 10 || Number.isInteger(kb) ? kb.toFixed(0) : kb.toFixed(1)} KB`;
  }
  const mb = size / (1024 * 1024);
  return `${mb >= 10 || Number.isInteger(mb) ? mb.toFixed(0) : mb.toFixed(1)} MB`;
}

function _artifactPublicBase() {
  // The server supplies the canonical public origin/path. Keep the relative
  // BASE fallback for local/dev deployments and existing embedded pages.
  if (typeof document !== "undefined" && typeof document.querySelector === "function") {
    const meta = document.querySelector('meta[name="sharedagents-public-base"]');
    const configured = String((meta && meta.content) || "").trim().replace(/\/+$/, "");
    if (configured) return configured;
  }
  return String(typeof BASE === "undefined" ? "" : (BASE || ""))
    .trim().replace(/\/+$/, "");
}

function _outputArtifactDescriptor(reference, sid) {
  if (!reference || typeof reference !== "object") return null;
  const digest = String(reference.sha256 || "").toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(digest)) return null;
  const sessionId = String(sid || "");
  if (!sessionId) return null;
  const mediaType = String(reference.media_type || "").toLowerCase();
  const isImage = OUTPUT_IMAGE_TYPES.has(mediaType);
  const name = String(
    reference.display_name || (isImage ? "Output image" : "Output file"),
  );
  const url =
    `${_artifactPublicBase()}/files/${encodeURIComponent(sessionId)}/${digest}/` +
    encodeURIComponent(name);
  return {
    digest,
    isImage,
    mediaType,
    typeLabel: OUTPUT_TYPE_LABELS.get(mediaType) || mediaType,
    name,
    size: _artifactSizeLabel(reference.size_bytes),
    url,
  };
}

function artifactOutputHtml(references, sid) {
  const seen = new Set();
  const items = [];
  for (const reference of Array.isArray(references) ? references : []) {
    const item = _outputArtifactDescriptor(reference, sid);
    if (!item || seen.has(item.digest)) continue;
    seen.add(item.digest);
    items.push(item);
  }
  if (!items.length) return "";

  const cards = [];
  for (const item of items) {
    const href = escapeAttr(item.url);
    const downloadHref = escapeAttr(item.url + "?download=1");
    const name = escapeHtml(item.name);
    const details = [item.typeLabel, item.size].filter(Boolean).join(" · ");
    if (item.isImage) {
      cards.push(`
        <figure class="output-artifact output-image">
          <button class="output-image-link" type="button"
                  data-artifact-url="${href}" data-artifact-name="${escapeAttr(item.name)}"
                  aria-label="Open ${escapeAttr(item.name)} fullscreen">
            <img src="${href}" alt="${escapeAttr(item.name)}" loading="lazy">
          </button>
          <figcaption>
            <span class="output-artifact-copy">
              <span class="output-artifact-name">${name}</span>
              <span class="output-artifact-meta" data-type-label="${escapeAttr(item.typeLabel)}"
                    data-size-label="${escapeAttr(item.size)}">${escapeHtml(details)}</span>
            </span>
            <a class="output-artifact-download" href="${downloadHref}" download title="Download ${escapeAttr(item.name)}">Download</a>
          </figcaption>
        </figure>`);
      continue;
    }
    const opensInViewer = item.mediaType === "application/pdf";
    const primaryHref = opensInViewer ? href : downloadHref;
    const primaryAttrs = opensInViewer
      ? 'target="_blank" rel="noopener noreferrer"'
      : "download";
    cards.push(`
      <article class="output-artifact output-file">
        <a class="output-file-primary" href="${primaryHref}" ${primaryAttrs}
           title="${opensInViewer ? "Open" : "Download"} ${escapeAttr(item.name)}">
          <span class="output-file-icon" aria-hidden="true">${fileIcon(item.name)}</span>
          <span class="output-artifact-copy">
            <span class="output-artifact-name">${name}</span>
            ${details ? `<span class="output-artifact-meta">${escapeHtml(details)}</span>` : ""}
          </span>
        </a>
        <a class="output-file-action" href="${downloadHref}" download
           aria-label="Download ${escapeAttr(item.name)}" title="Download ${escapeAttr(item.name)}">Download</a>
      </article>`);
  }
  const countLabel = `${items.length} attachment${items.length === 1 ? "" : "s"}`;
  return `
    <div class="output-artifact-heading">
      <span>Attachments</span>
      <span class="output-artifact-count" aria-label="${countLabel}">${items.length}</span>
    </div>
    <div class="output-artifact-grid${items.length > 1 ? " has-multiple" : ""}">
      ${cards.join("")}
    </div>`;
}

function _updateArtifactImageMeta(img) {
  if (!img || !img.naturalWidth || !img.naturalHeight) return;
  const card = img.closest(".output-image");
  const meta = card && card.querySelector(".output-artifact-meta");
  if (!meta) return;
  const parts = [
    meta.dataset.typeLabel || "",
    `${img.naturalWidth}×${img.naturalHeight}`,
    meta.dataset.sizeLabel || "",
  ].filter(Boolean);
  meta.textContent = parts.join(" · ");
}

let _artifactLightbox = null;
let _artifactLightboxOpener = null;

function _closeArtifactLightbox() {
  if (!_artifactLightbox || _artifactLightbox.hidden) return;
  _artifactLightbox.hidden = true;
  _artifactLightbox.querySelector("img").removeAttribute("src");
  document.body.classList.remove("artifact-lightbox-open");
  if (_artifactLightboxOpener) _artifactLightboxOpener.focus();
  _artifactLightboxOpener = null;
}

function _ensureArtifactLightbox() {
  if (_artifactLightbox || typeof document === "undefined") return _artifactLightbox;
  const overlay = document.createElement("div");
  overlay.className = "artifact-lightbox";
  overlay.hidden = true;
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-modal", "true");
  overlay.setAttribute("aria-labelledby", "artifact-lightbox-title");
  overlay.innerHTML = `
    <div class="artifact-lightbox-panel">
      <header class="artifact-lightbox-header">
        <span id="artifact-lightbox-title"></span>
        <button type="button" class="artifact-lightbox-close" aria-label="Close fullscreen image">×</button>
      </header>
      <div class="artifact-lightbox-body"><img alt=""></div>
      <footer class="artifact-lightbox-footer">
        <span class="artifact-lightbox-meta"></span>
        <a class="artifact-lightbox-download" download>Download original</a>
      </footer>
    </div>`;
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay || event.target.closest(".artifact-lightbox-close")) {
      _closeArtifactLightbox();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !overlay.hidden) {
      event.preventDefault();
      event.stopImmediatePropagation();
      _closeArtifactLightbox();
    }
  }, true);
  document.body.appendChild(overlay);
  _artifactLightbox = overlay;
  return overlay;
}

/** Open the shared fullscreen image lightbox.
 *  Accepts either a button element with data-artifact-url/name, or an
 *  explicit {url, name, opener} descriptor (user thumbs / paste preview).
 */
function _openArtifactLightbox(buttonOrOpts) {
  const overlay = _ensureArtifactLightbox();
  if (!overlay || !buttonOrOpts) return;
  let url = "";
  let name = "Image";
  let opener = null;
  if (typeof buttonOrOpts === "object" && buttonOrOpts.dataset) {
    opener = buttonOrOpts;
    url = buttonOrOpts.dataset.artifactUrl || "";
    name = buttonOrOpts.dataset.artifactName || "Output image";
  } else if (typeof buttonOrOpts === "object") {
    url = buttonOrOpts.url || "";
    name = buttonOrOpts.name || "Image";
    opener = buttonOrOpts.opener || null;
  }
  if (!url) return;
  _artifactLightboxOpener = opener;
  overlay.querySelector("#artifact-lightbox-title").textContent = name;
  const img = overlay.querySelector("img");
  img.alt = name;
  img.onload = () => {
    overlay.querySelector(".artifact-lightbox-meta").textContent =
      `${img.naturalWidth}×${img.naturalHeight}`;
  };
  img.src = url;
  const download = overlay.querySelector(".artifact-lightbox-download");
  // data:/blob: cannot take ?download=1 — use the raw URL; authenticated
  // /files routes get the download query the asset route already understands.
  if (/^(data:|blob:)/i.test(url)) {
    download.href = url;
  } else {
    download.href = url + (url.includes("?") ? "&" : "?") + "download=1";
  }
  download.setAttribute("download", name);
  overlay.hidden = false;
  document.body.classList.add("artifact-lightbox-open");
  overlay.querySelector(".artifact-lightbox-close").focus();
}

/** Alias used by paste-preview / user-thumb click paths. */
function openMediaLightbox(url, name, opener) {
  _openArtifactLightbox({ url, name: name || "Image", opener: opener || null });
}

function _bindArtifactOutputUi(container) {
  if (!container) return;
  container.querySelectorAll(".output-image img").forEach((img) => {
    if (img.complete) _updateArtifactImageMeta(img);
    else img.addEventListener("load", () => _updateArtifactImageMeta(img), { once: true });
  });
  container.querySelectorAll(".output-image-link").forEach((button) => {
    button.addEventListener("click", () => _openArtifactLightbox(button));
  });
  // User-attached image thumbs reuse the same lightbox.
  container.querySelectorAll("button.user-image-link").forEach((button) => {
    if (button.dataset.lightboxBound === "1") return;
    button.dataset.lightboxBound = "1";
    button.addEventListener("click", () => _openArtifactLightbox(button));
  });
}

function _fmtTurnElapsed(ms) {
  const s = Math.max(0, Math.floor(Number(ms) / 1000));
  if (s < 60) return s + "s";
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
}

function _turnFooterTimes(messageEl) {
  const idx = state.msgCache
    ? state.msgCache.findIndex((m) => m && m.el === messageEl)
    : -1;
  const cached = idx >= 0 ? state.msgCache[idx] : null;
  const started = (cached && cached.timestamp)
    || (messageEl && messageEl.dataset && messageEl.dataset.turnStarted)
    || "";
  const ended = (cached && cached.endedAt)
    || (messageEl && messageEl.dataset && messageEl.dataset.turnEnded)
    || "";
  const status = (cached && cached.status)
    || (messageEl && messageEl.dataset && messageEl.dataset.turnStatus)
    || "";
  const terminal = status === "ok"
    || status === "aborted"
    || status === "error";
  const startMs = Date.parse(started);
  if (ended) {
    // Recorded completion exists: label + FROZEN elapsed from the real end.
    // A finished turn may still show running for an unfinished tool;
    // once endedAt is stamped, freeze elapsed — do not keep ticking into the
    // next turn (Status 20:21 + flush 20:38 painted as 17:05).
    const endMs = Date.parse(ended);
    const elapsedMs = Number.isFinite(startMs) && Number.isFinite(endMs)
      ? Math.max(0, endMs - startMs)
      : null;
    return { started, ended, last: ended, status, elapsedMs };
  }
  if (terminal) {
    // A COMPLETED turn whose recorded completion stamp is absent (historical
    // row that never carried ended_at, or a seal with no event ts): degrade.
    // Never substitute the current time — no time label and no elapsed.
    return { started, ended: "", last: "", status, elapsedMs: null };
  }
  // Live / unknown card: while the turn is genuinely RUNNING the ticker may
  // show a live elapsed (a real count-up, not a completed-turn claim), and
  // the label comes from the event-derived lastTs only — never synthesized.
  const last = (cached && cached.lastTs)
    || (status === "running" ? started : "");
  const endNow = startMs && Number.isFinite(startMs) && status === "running"
    ? Date.now()
    : NaN;
  const elapsedMs = Number.isFinite(startMs) && Number.isFinite(endNow)
    ? Math.max(0, endNow - startMs)
    : null;
  return { started, ended: "", last, status, elapsedMs };
}

let _turnFooterTimer = null;
function _ensureTurnFooterTicker() {
  if (_turnFooterTimer) return;
  _turnFooterTimer = setInterval(() => {
    const running = document.querySelectorAll(
      ".message-assistant.message-running",
    );
    if (!running.length) {
      clearInterval(_turnFooterTimer);
      _turnFooterTimer = null;
      return;
    }
    running.forEach((el) => renderMessageTelemetry(el));
  }, 250);
}

function renderMessageTelemetry(messageEl) {
  if (!messageEl || !messageEl.querySelector) return;
  const host = messageEl.querySelector(".message-telemetry");
  if (!host) return;
  const freshMetrics = Array.from(
    messageEl.querySelectorAll(".message-content .assist-metrics.block-metrics"),
  );
  const metrics = freshMetrics.length ? freshMetrics : Array.from(
    host.querySelectorAll(".turn-details-body .assist-metrics.block-metrics"),
  );
  const openDetails = host.querySelector("details.turn-details");
  const wasOpen = !!(openDetails && openDetails.open);
  const times = _turnFooterTimes(messageEl);
  const pill = typeof _historyTurnPillHtml === "function"
    ? _historyTurnPillHtml(times.status)
    : "";
  const lastLabel = typeof formatMessageTime === "function"
    ? formatMessageTime(times.last)
    : "";
  // elapsedMs is null when no meaningful end boundary exists (completed turn
  // without a recorded stamp) — omit the duration then, never print 0s.
  const elapsed = times.started && times.elapsedMs != null
    ? _fmtTurnElapsed(times.elapsedMs)
    : "";
  host.hidden = false;
  const footer = document.createElement("div");
  footer.className = "turn-footer";
  const meta = document.createElement("span");
  meta.className = "turn-status-meta";
  if (pill) meta.insertAdjacentHTML("beforeend", pill);
  if (lastLabel) {
    const ts = document.createElement("span");
    ts.className = "turn-last-ts";
    ts.textContent = lastLabel;
    meta.appendChild(ts);
  }
  if (elapsed) {
    const dur = document.createElement("span");
    dur.className = "turn-elapsed";
    dur.textContent = elapsed;
    meta.appendChild(dur);
  }
  const slot = document.createElement("span");
  slot.className = "turn-details-slot";
  if (metrics.length || times.status === "aborted") {
    const details = document.createElement("details");
    details.className = "turn-details";
    if (wasOpen) details.open = true;
    const summary = document.createElement("summary");
    summary.textContent = "Turn details";
    const body = document.createElement("div");
    body.className = "turn-details-body";
    metrics.forEach((metric) => body.appendChild(metric));
    if (!metrics.length) {
      const note = document.createElement("p");
      note.className = "turn-details-empty";
      note.textContent = "This turn was aborted before usage metrics were reported.";
      body.appendChild(note);
    }
    details.append(summary, body);
    slot.appendChild(details);
  }
  footer.append(meta, slot);
  host.replaceChildren(footer);
  if (times.status === "running") _ensureTurnFooterTicker();
}

function renderMessageArtifacts(messageEl, references, sid) {
  if (!messageEl) return;
  const container = messageEl.querySelector(".message-artifacts");
  if (!container) return;
  const html = artifactOutputHtml(references, sid);
  container.innerHTML = html;
  container.hidden = !html;
  messageEl.classList.toggle("has-artifacts", !!html);
  if (html) _bindArtifactOutputUi(container);
  renderMessageTelemetry(messageEl);
}

function appendLiveArtifact(turn, reference) {
  if (!turn || !turn.el) return;
  const item = _outputArtifactDescriptor(reference, state.currentSessionId);
  if (!item) return;
  if (!Array.isArray(turn.artifacts)) turn.artifacts = [];
  if (!turn.artifacts.some((ref) => ref.sha256 === item.digest)) {
    turn.artifacts.push({
      sha256: item.digest,
      display_name: item.name,
      media_type: item.mediaType,
      size_bytes: Number(reference.size_bytes || 0),
    });
  }
  renderMessageArtifacts(turn.el, turn.artifacts, state.currentSessionId);
  const idx = state.msgCache.findIndex((message) => message.el === turn.el);
  if (idx >= 0) state.msgCache[idx].artifacts = turn.artifacts.slice();
}

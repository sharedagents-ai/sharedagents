/* polish.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */
// ══════════════════════════════════════════════════════════════════════
// open-webui-style polish — additive, isolated module. Runs at script load
// (app.js is loaded at end of <body>, so the DOM is parsed). Everything is
// event-delegated or MutationObserver-based so it never depends on init()
// ordering and can't break the existing chat logic.
// ══════════════════════════════════════════════════════════════════════
// Live ticker tone. Working phases stay gray; red is a real error line.
// Color emoji (🔧 🌀 ✍️) ignore CSS color and look like failures.
function statusTextTone(text) {
  const t = String(text || "").trim();
  if (!t) return "";
  if (
    /\b(?:prefill|reasoning|warming|writing|sending|running|compacting|prompt received|steering|waiting|thinking|still working|queued)\b/i.test(
      t,
    )
  ) {
    return "thinking";
  }
  if (/(?:^|\b)error\b/i.test(t) || /^gateway unreachable$/i.test(t)) {
    return "error";
  }
  return "";
}

(function owPolish() {
  "use strict";

  // ── Loop 4: per-code-block copy buttons (open-webui staple) ──────────
  function enhanceCodeBlocks(root) {
    (root || document)
      .querySelectorAll(".message-content pre")
      .forEach((pre) => {
        if (pre.dataset.owCopy) return;
        pre.dataset.owCopy = "1";
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "code-copy-btn";
        btn.textContent = "Copy";
        btn.setAttribute("aria-label", "Copy code");
        btn.addEventListener("click", async (e) => {
          e.stopPropagation();
          const code = pre.querySelector("code");
          const txt = (code ? code.innerText : pre.innerText) || "";
          try {
            await navigator.clipboard.writeText(txt);
          } catch (_) {
            const ta = document.createElement("textarea");
            ta.value = txt;
            document.body.appendChild(ta);
            ta.select();
            try {
              document.execCommand("copy");
            } catch (__) {}
            ta.remove();
          }
          btn.textContent = "Copied";
          btn.classList.add("copied");
          setTimeout(() => {
            btn.textContent = "Copy";
            btn.classList.remove("copied");
          }, 1400);
        });
        pre.appendChild(btn);
      });
  }
  const messages = document.getElementById("messages");
  if (messages) {
    enhanceCodeBlocks(messages);
    const mo = new MutationObserver((muts) => {
      for (const m of muts)
        if (m.addedNodes.length) {
          enhanceCodeBlocks(messages);
          break;
        }
    });
    mo.observe(messages, { childList: true, subtree: true });
  }
})();

// ── Loop 5: sidebar conversation search/filter (open-webui staple) ─────
(function owConvSearch() {
  "use strict";
  const list = document.getElementById("session-list");
  if (!list || document.getElementById("conv-search")) return;
  const wrap = document.createElement("div");
  wrap.className = "conv-search-wrap";
  wrap.innerHTML =
    '<svg class="conv-search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>' +
    '<input id="conv-search" type="search" placeholder="Search chats…" aria-label="Search conversations" autocomplete="off">';
  list.parentNode.insertBefore(wrap, list);
  const input = wrap.querySelector("#conv-search");
  function apply() {
    const q = input.value.trim().toLowerCase();
    let shown = 0;
    list.querySelectorAll(":scope > *").forEach((row) => {
      if (row.classList && row.classList.contains("conv-empty")) {
        row.remove();
        return;
      }
      const hit = !q || (row.textContent || "").toLowerCase().includes(q);
      row.style.display = hit ? "" : "none";
      if (hit) shown++;
    });
    let empty = list.parentNode.querySelector(".conv-empty");
    if (q && shown === 0) {
      if (!empty) {
        empty = document.createElement("div");
        empty.className = "conv-empty";
        list.parentNode.appendChild(empty);
      }
      empty.textContent = `No chats matching “${input.value.trim()}”`;
    } else if (empty) {
      empty.remove();
    }
  }
  input.addEventListener("input", apply);
  // Re-apply when the list re-renders (new chat / refresh).
  new MutationObserver(() => {
    if (input.value) apply();
  }).observe(list, { childList: true });
  // Ctrl/Cmd+K focuses search (open-webui shortcut).
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K")) {
      e.preventDefault();
      if (window.MenuState) MenuState.setSidebarVisible(true);
      else if (document.body.classList) document.body.classList.add("sidebar-open");
      input.focus();
      input.select();
    }
  });
})();

// ── Loop 6: welcome-screen suggestion chips (open-webui example prompts) ─
(function owSuggestions() {
  "use strict";
  const wc = document.querySelector("#welcome-screen .welcome-content");
  const input = document.getElementById("message-input");
  if (!wc || !input || wc.querySelector(".welcome-suggestions")) return;
  const PROMPTS = [
    "What can you do?",
    "Summarize the latest news on a topic I care about",
    "Research a question and write it up",
    "/help",
  ];
  const box = document.createElement("div");
  box.className = "welcome-suggestions";
  PROMPTS.forEach((p) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "welcome-chip";
    chip.textContent = p;
    chip.addEventListener("click", () => {
      input.value = p;
      input.dispatchEvent(new Event("input", { bubbles: true })); // enable send + autogrow
      input.focus();
    });
    box.appendChild(chip);
  });
  wc.appendChild(box);
})();

// ── Loop 8: "thinking" pulse — make the live status read as active (typing) ─
(function owThinking() {
  "use strict";
  const st = document.getElementById("status-text");
  if (!st) return;
  const upd = () => {
    const tone = statusTextTone(st.textContent || "");
    // Voice used to set style.color = var(--error) and leave it on the
    // next ticker line (prefill/reasoning). Refresh cleared the inline
    // style — that is why the same page looked gray after reload.
    st.style.color = "";
    st.classList.toggle("is-error", tone === "error");
    st.classList.toggle("is-thinking", tone === "thinking");
  };
  new MutationObserver(upd).observe(st, {
    childList: true,
    characterData: true,
    subtree: true,
  });
  upd();
})();

// ── Loop 9: model-picker mobile label — drop the " · <ctx>" suffix on narrow
// screens so the native select shows "local" instead of the truncated "· ·".
// Value is never touched; full label kept as a tooltip + restored on widen.
(function owModelLabel() {
  "use strict";
  const sel = document.getElementById("model-select");
  if (!sel) return;
  const mq = window.matchMedia("(max-width: 768px)");
  function apply() {
    sel.querySelectorAll("option").forEach((o) => {
      if (o.dataset.full === undefined) o.dataset.full = o.textContent;
      if (mq.matches) {
        o.textContent = o.dataset.full.split("·")[0].trim();
        o.title = o.dataset.full;
      } else {
        o.textContent = o.dataset.full;
        o.removeAttribute("title");
      }
    });
  }
  apply();
  new MutationObserver(apply).observe(sel, { childList: true });
  try {
    mq.addEventListener("change", apply);
  } catch (_) {
    mq.addListener && mq.addListener(apply);
  }
})();

// ── Loop 10: friendly empty-state when there are no conversations yet ──
(function owEmptyChats() {
  "use strict";
  const list = document.getElementById("session-list");
  if (!list) return;
  function upd() {
    const rows = list.querySelectorAll(":scope > .session-item").length;
    let ph = list.parentNode.querySelector(".chats-empty-state");
    if (rows === 0 && !document.querySelector(".conv-empty")) {
      if (!ph) {
        ph = document.createElement("div");
        ph.className = "chats-empty-state";
        ph.innerHTML =
          '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg><div class="ces-title">No conversations yet</div><div class="ces-sub">Press <b>New&nbsp;Chat</b> to start.</div>';
        list.parentNode.appendChild(ph);
      }
    } else if (ph) {
      ph.remove();
    }
  }
  new MutationObserver(upd).observe(list, { childList: true });
  upd();
})();

// ── Loop 11: keyboard shortcut — Ctrl/⌘+Shift+C copies the last assistant answer ─
(function owCopyLast() {
  "use strict";
  document.addEventListener("keydown", (e) => {
    if (
      (e.ctrlKey || e.metaKey) &&
      e.shiftKey &&
      (e.key === "c" || e.key === "C")
    ) {
      const nodes = document.querySelectorAll(
        "#messages .message.assistant .message-content",
      );
      const last = nodes[nodes.length - 1];
      if (!last) return;
      e.preventDefault();
      navigator.clipboard.writeText(last.innerText || "").catch(() => {});
      const st = document.getElementById("status-text");
      if (st) {
        const o = st.textContent;
        st.textContent = "Copied last answer";
        setTimeout(() => {
          if (st.textContent === "Copied last answer") st.textContent = o;
        }, 1200);
      }
    }
  });
})();

// ── Loop 14: surface keyboard shortcuts inside Settings ──
(function owSettingsTip() {
  "use strict";
  const mb = document.querySelector("#settings-modal .modal-body");
  if (!mb || mb.querySelector(".settings-tip")) return;
  const d = document.createElement("div");
  d.className = "settings-tip";
  d.innerHTML =
    '<div class="settings-tip-h">Keyboard</div><ul>' +
    "<li><kbd>⌘/Ctrl</kbd>+<kbd>K</kbd> — search chats</li>" +
    "<li><kbd>⌘/Ctrl</kbd>+<kbd>⇧</kbd>+<kbd>C</kbd> — copy last answer</li>" +
    "<li><kbd>/</kbd> commands · <kbd>↵</kbd> send · <kbd>⇧</kbd>+<kbd>↵</kbd> newline</li></ul>";
  mb.appendChild(d);
})();

// Keyboard occlusion: lift the composer by the visualViewport gap and hide
// Agent Monitor while the OS keyboard covers the bottom of the screen.
(function kbViewport() {
  "use strict";
  const root = document.documentElement;
  const apply = () => {
    const vv = window.visualViewport;
    let occluded = 0;
    if (vv) {
      occluded = Math.max(0, window.innerHeight - vv.height - vv.offsetTop);
    }
    root.style.setProperty("--kb-inset", occluded + "px");
    document.body.classList.toggle("kb-open", occluded > 80);
    if (typeof resizeComposer === "function") resizeComposer();
  };
  if (window.visualViewport) {
    window.visualViewport.addEventListener("resize", apply);
    window.visualViewport.addEventListener("scroll", apply);
  }
  window.addEventListener("focusin", apply);
  window.addEventListener("focusout", () => setTimeout(apply, 50));
  apply();
})();

/* send.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */
function normalizeQueueIds(queue_ids) {
  const values = Array.isArray(queue_ids) ? queue_ids : [queue_ids];
  return [...new Set(values.map((id) => String(id || "")).filter(Boolean))];
}

function queueItemsForIds(queue_ids) {
  const wanted = new Set(normalizeQueueIds(queue_ids));
  return (state.queue || []).filter((item) => wanted.has(String(item.queue_id || "")));
}

function queueItemAlreadyDelivered(error) {
  const data = (error && error.data) || {};
  return !!(error && error.status === 409 &&
    (data.error_code || data.code) === "not_pending");
}

function preserveQueueDraft(text) {
  const draft = messageInput.value;
  messageInput.value = draft && draft !== text ? draft + "\n\n" + text : text;
  updateSendState();
}

async function queueDeleteMessages(queue_ids) {
  const sid = state.currentSessionId;
  const ids = normalizeQueueIds(queue_ids);
  if (!sid || !ids.length) return false;
  const wanted = new Set(ids);
  // Optimistic remove; queue_msg_removed events reconcile successful deletes.
  state.queue = state.queue.filter((q) => !wanted.has(String(q.queue_id || "")));
  renderHarnessWidget();
  try {
    for (const queueId of ids) {
      try {
        await apiDelete(
          '/api/sessions/' + encodeURIComponent(sid) +
          '/queue/' + encodeURIComponent(queueId)
        );
      } catch (error) {
        if (!queueItemAlreadyDelivered(error)) throw error;
      }
    }
    return true;
  } catch (e) {
    console.error("queueDeleteMessages failed:", e);
    if (state.currentSessionId === sid && typeof refreshSessionWidgets === "function") {
      await refreshSessionWidgets();
    }
    return false;
  }
}

async function queueDelete(queue_id) {
  return queueDeleteMessages([queue_id]);
}

async function queueEditMessages(queue_ids, currentText) {
  const newText = await promptCard({
    title: "Edit queued message",
    value: currentText,
    okLabel: "Save",
  });
  if (newText === null || newText === currentText) return false;
  if (!String(newText).trim()) return false;
  const sid = state.currentSessionId;
  const ids = normalizeQueueIds(queue_ids);
  const items = queueItemsForIds(ids);
  if (!sid || !items.length) return false;
  const primary = items[0];
  const extras = items.slice(1);
  const removed = [];
  let patched = false;
  let editingId = primary.queue_id;
  try {
    const updated = await apiPatch(
      '/api/sessions/' + encodeURIComponent(sid) +
      '/queue/' + encodeURIComponent(primary.queue_id),
      { text: newText, queue_ids: ids }
    );
    patched = true;
    for (const item of extras) {
      editingId = item.queue_id;
      await apiDelete(
        '/api/sessions/' + encodeURIComponent(sid) +
        '/queue/' + encodeURIComponent(item.queue_id)
      );
      removed.push(item);
    }
    if (state.currentSessionId === sid) {
      const mergedIds = new Set(ids);
      state.queue = state.queue.flatMap((item) => {
        if (item.queue_id === primary.queue_id) return [{ ...item, ...(updated?.item ?? {text: newText, display_text: newText}) }];
        return mergedIds.has(String(item.queue_id || "")) ? [] : [item];
      });
      renderHarnessWidget();
    }
    return true;
  } catch (e) {
    const delivered = queueItemAlreadyDelivered(e);
    if (delivered && state.currentSessionId === sid) {
      state.queue = state.queue.filter((item) => item.queue_id !== editingId);
      preserveQueueDraft(newText);
      appendSystemMessage("The message was already delivered. Your edit is in the composer.", true);
      renderHarnessWidget();
    }
    if (!delivered) console.error("queueEditMessages failed:", e);
    // A multi-item edit is implemented as update-first + remove-rest. Roll
    // back completed mutations when a later request fails so text is not lost.
    if (patched && extras.length) {
      try {
        await apiPatch(
          '/api/sessions/' + encodeURIComponent(sid) +
          '/queue/' + encodeURIComponent(primary.queue_id),
          { text: primary.text }
        );
      } catch (rollbackError) {
        console.error("Could not roll back merged queue edit:", rollbackError);
      }
    }
    for (const item of removed) {
      try {
        await apiPost(
          '/api/sessions/' + encodeURIComponent(sid) + '/queue',
          { text: item.text, source: item.source || "", author: item.author || "" },
        );
      } catch (restoreError) {
        console.error("Could not restore queue item after edit:", restoreError);
      }
    }
    if (state.currentSessionId === sid && typeof refreshSessionWidgets === "function") {
      await refreshSessionWidgets();
    }
    return false;
  }
}

async function queueEdit(queue_id, currentText) {
  return queueEditMessages([queue_id], currentText);
}

function clientRequestId() {
  return crypto.randomUUID ? crypto.randomUUID() :
    Array.from(crypto.getRandomValues(new Uint8Array(16)), (v) => v.toString(16).padStart(2, "0")).join("");
}

// Acceptance comes from a command receipt or its canonical turn_started echo.
function acceptPromptReceipt(pending) {
  if (!pending || pending.sid !== state.currentSessionId) return;
  pending.accepted = true;
  if (statusText.textContent === "Submission pending…") statusText.textContent = "Prompt received";
  if (messageInput.value.trim() === pending.draft.trim()) {
    messageInput.value = "";
    messageInput.style.height = "auto";
  }
  if (pending.images && state.pastedImages === pending.images) state.pastedImages = [];
  if (pending.files && state.attachedTextFiles === pending.files) state.attachedTextFiles = [];
  if (typeof renderPastePreview === "function") renderPastePreview();
  if (state.pendingPrompt === pending) state.pendingPrompt = null;
  updateSendState();
}

function watchPromptReceipt(pending, path, payload, onAccepted, onRejected) {
  setTimeout(async () => {
    if (pending.accepted || state.pendingPrompt !== pending || state.currentSessionId !== pending.sid) return;
    try {
      const receipt = await apiPost(path, payload); // same id, idempotent
      if (pending.accepted || state.currentSessionId !== pending.sid) return;
      if (receipt.pending) watchPromptReceipt(pending, path, payload, onAccepted, onRejected);
      else if (receipt.accepted) {
        acceptPromptReceipt(pending);
        onAccepted(receipt);
      } else throw new Error(receipt.error || "Prompt rejected");
    } catch (error) {
      if (pending.accepted || state.currentSessionId !== pending.sid) return;
      if (state.pendingPrompt === pending) state.pendingPrompt = null;
      onRejected(error);
      updateSendState();
    }
  }, 1000);
}

function reconcilePromptReceipt(event) {
  const pending = state.pendingPrompt;
  if (!pending || pending.sid !== state.currentSessionId) return;
  const parts = _userPromptParts(event.user);
  if (event.request_id === pending.requestId || _pendingEchoMatches(
    pending.echo, pending.sid, parts.text, parts.images.length,
  )) acceptPromptReceipt(pending);
}

function waitForCancelSettlement(timeoutMs = 1500) {
  return new Promise((resolve) => {
    const deadline = Date.now() + timeoutMs;
    const check = () => {
      if (!state.pendingCancel) return resolve(true);
      // A timeout is unknown, never evidence that a stop completed.
      if (Date.now() >= deadline) return resolve(false);
      setTimeout(check, 50);
    };
    check();
  });
}

async function restoreSteeredQueueItem(sid, item) {
  const res = await apiPost(
    '/api/sessions/' + encodeURIComponent(sid) + '/queue',
    {
      text: item.text,
      ...(item.texts ? {texts: item.texts} : {}),
      source: item.source || "web",
      author: item.author || "",
    },
  );
  if (state.currentSessionId === sid &&
      !(state._renderedInjects && state._renderedInjects.has(res.queue_id))) {
    // The SSE receipt may beat the POST response. Replace rather than append
    // so a restored item cannot appear twice or return after delivery.
    const replacedIds = new Set([item.queue_id, res.queue_id]);
    state.queue = state.queue.filter((q) => !replacedIds.has(q.queue_id));
    state.queue.push({
      ...item,
      ...res.item,
      queue_id: res.queue_id || item.queue_id,
    });
    renderHarnessWidget();
  }
}

// Keep the normal Web path as one durable queue entry: once the first item is
// acknowledged, later drafts append to it with a single PATCH. Entries from
// other clients keep their own receipt/provenance and are aggregated by the UI.
//
// Returns the SERVER's verdict, because the composer must not claim a queue
// entry the gateway never wrote (it used to treat `accepted` as queued, so a
// follow-up that actually cancelled the live turn still showed "queued ✓"):
//   "queued"   — appended to the durable follow-up queue
//   "new-turn" — accepted, but as a NEW turn (nothing to queue behind)
//   ""         — not accepted
async function enqueueQueuedMessage(sid, text, draftText = text, files = []) {
  const draft = String(text || "").trim();
  if (!sid || (!draft && !(files && files.length))) return "";
  const pending = {
    sid, requestId: clientRequestId(), draft: draftText, accepted: false,
    images: state.pastedImages, files: state.attachedTextFiles,
    echo: { sid, text: draft, echoText: draftText, images: 0 },
  };
  state.pendingPrompt = pending;
  const path = '/api/sessions/' + encodeURIComponent(sid) + '/prompt';
  const payload = { prompt: draft, frontend: "web", request_id: pending.requestId };
  if (files && files.length) payload.files = files;
  try {
    const res = await apiPost(path, payload);
    if (res && res.pending) {
      if (!pending.accepted && state.currentSessionId === sid) {
        statusText.textContent = "Submission pending…";
        watchPromptReceipt(pending, path, payload, () => {}, (error) => {
          appendSystemMessage(`Queue error: ${escapeHtml(error.message)}`, true);
        });
        if (_liveStreamNeedsOpen(sid)) openLiveStream(sid, 0);
      }
      return "pending";
    }
    if (state.pendingPrompt === pending) state.pendingPrompt = null;
    if (res && res.queued === true) return "queued";
    if (pending.accepted || (res && res.accepted === true)) return "new-turn";
    return "";
  } catch (error) {
    if (state.pendingPrompt === pending) state.pendingPrompt = null;
    if (pending.accepted) return "new-turn";
    throw error;
  }
}

// Remove a prompt bubble painted before the gateway accepted it.
function dropOptimisticUserEcho(el) {
  if (!el) return;
  try { el.remove(); } catch (_e) {}
  if (state.msgCache && Array.isArray(state.msgCache)) {
    const idx = state.msgCache.findIndex((m) => m && m.el === el);
    if (idx >= 0) state.msgCache.splice(idx, 1);
  }
}

// Paint the steered prompt as the user bubble and register it as the pending
// echo, so the turn_started that follows does not paint it a second time.
function paintOptimisticUserEcho(sid, text, files = []) {
  const el = appendMessage(
    "user",
    text,
    true,
    files,
    null,
    null,
    new Date().toISOString(),
    viewerAuthor(state.me),
    { preserveLiveTurn: true },
  );
  state.pendingEcho = { sid, text, echoText: text, images: 0, el };
  if (state.scroll) state.scroll.autoScroll = true;
  if (typeof VirtualScroller !== "undefined") {
    VirtualScroller._hideNewMsgBtn();
    VirtualScroller._forcePin();
  }
  return el;
}

// Ask the server to steer. It decides from its own view of the session, not
// this page's idle flag: a running turn that can take the queue at once (every
// ACP agent) gets it now and answers `steered: true`; otherwise `busy` says
// whether the turn must be stopped before the text is sent as a new prompt.
// Returns that reply, "delivered" when the turn already took the items, or
// null when the route is unavailable (the page then falls back to its flag).
async function requestServerSteer(sid, ids) {
  try {
    return await apiPost(
      '/api/sessions/' + encodeURIComponent(sid) + '/queue/steer',
      { queue_ids: ids },
    );
  } catch (error) {
    if (queueItemAlreadyDelivered(error)) return "delivered";
    console.error("Server steer unavailable:", error);
    return null;
  }
}

// Steer the visible queued text. A running turn that can take it in place
// does so (server-side). Otherwise stop the current execution and promote all
// visible queued text to one normal prompt. Removing every durable item before
// cancel is intentional: turn-end auto-drain must not answer any part a second
// time.
async function steerQueueMessages(queue_ids) {
  if (state.monitorReadOnly || state.steeringQueueId) return false;
  const sid = state.currentSessionId;
  const ids = normalizeQueueIds(queue_ids);
  const items = queueItemsForIds(ids);
  if (!sid || !items.length) return false;
  let text = "";
  let displayText = "";
  let attachedFiles = [];
  const lockId = String(items[0].queue_id || "");

  state.steeringQueueId = lockId;
  renderHarnessWidget();
  const removedItems = [];
  let accepted = false;
  let optimisticEchoEl = null;
  try {
    const steer = await requestServerSteer(sid, ids);
    if (steer === "delivered" || (steer && steer.steered === true)) {
      // The running turn takes the items; its queue_msg_injected events
      // remove them from the queue and paint them in the transcript.
      return true;
    }
    const serverBusy = !!(steer && steer.busy === true);
    const prepared = await apiPost(
      '/api/sessions/' + encodeURIComponent(sid) + '/queue/prepare',
      { queue_ids: ids },
    );
    text = prepared.text;
    displayText = prepared.display_text ?? text;
    attachedFiles = prepared.attached_files || [];
    let delivered = false;
    for (const item of items) {
      try {
        await apiDelete(
          '/api/sessions/' + encodeURIComponent(sid) +
          '/queue/' + encodeURIComponent(item.queue_id),
        );
      } catch (error) {
        // The gateway answers 409 not_pending once the running turn has
        // already taken the message (it is delivered at the agent's next
        // step). There is nothing left to steer and no failure to report.
        if (!queueItemAlreadyDelivered(error)) throw error;
        if (state.currentSessionId === sid) {
          state.queue = state.queue.filter((q) => q.queue_id !== item.queue_id);
        }
        delivered = true;
        break;
      }
      removedItems.push(item);
      if (state.currentSessionId === sid) {
        state.queue = state.queue.filter((q) => q.queue_id !== item.queue_id);
      }
    }
    if (delivered) {
      if (removedItems.length) {
        // Parts removed before the delivered one go back to the queue.
        const restoredItem = { ...removedItems[0], text: mergeQueuedText(removedItems), texts: removedItems.map(item => item.text) };
        removedItems.length = 0;
        try {
          await restoreSteeredQueueItem(sid, restoredItem);
        } catch (restoreError) {
          console.error("Could not restore steered queue items:", restoreError);
          if (state.currentSessionId === sid) {
            preserveQueueDraft(restoredItem.text);
            appendSystemMessage("Could not restore the removed queue text. It is in the composer.", true);
          }
          return false;
        }
      }
      return true;
    }
    if (state.currentSessionId === sid) renderHarnessWidget();

    const busy = serverBusy || (typeof treeBusy === "function"
      ? treeBusy()
      : !!state.isStreaming);
    // Paint the transport echo without sealing the live assistant: only its
    // backend terminal event can decide whether it was aborted.
    const cancelRequest = busy ? cancelStreaming({ awaitServer: true }) : null;
    optimisticEchoEl = paintOptimisticUserEcho(sid, displayText, attachedFiles);
    if (cancelRequest) {
      await cancelRequest;
      await waitForCancelSettlement(1500);
    }
    if (state.currentSessionId !== sid) {
      throw new Error("the session changed before the steered message was sent");
    }

    messageInput.value = displayText;
    messageInput.style.height = "auto";
    updateSendState();
    accepted = (await sendMessage({
      skipPendingAskSteer: true,
      skipBusyQueue: true,
      optimisticEchoEl,
      queuePrompt: text,
    })) === true;
    if (!accepted) throw new Error("the steered message was not sent");
    return true;
  } catch (e) {
    if (optimisticEchoEl) {
      dropOptimisticUserEcho(optimisticEchoEl);
      if (state.pendingEcho && state.pendingEcho.el === optimisticEchoEl) {
        state.pendingEcho = null;
      }
    }
    // A failed delete, cancel, or prompt must leave already-removed text
    // recoverable. Restore that subset as one item before surfacing the error.
    if (removedItems.length && !accepted) {
      const restoredItem = {
        ...removedItems[0],
        text: mergeQueuedText(removedItems),
        texts: removedItems.map(item => item.text),
      };
      try {
        await restoreSteeredQueueItem(sid, restoredItem);
      } catch (restoreError) {
        console.error("Could not restore steered queue items:", restoreError);
        if (state.currentSessionId === sid) {
          preserveQueueDraft(restoredItem.text);
        }
      }
    }
    if (state.currentSessionId === sid) {
      appendSystemMessage(`Steer error: ${escapeHtml(e.message)}`, true);
    }
    return false;
  } finally {
    if (state.steeringQueueId === lockId) state.steeringQueueId = "";
    if (state.currentSessionId === sid) renderHarnessWidget();
  }
}

async function steerQueueMessage(queue_id) {
  return steerQueueMessages([queue_id]);
}

// ── Chat ──

function applyFirstSubmitSession(response) {
  if (!response || !(response.session_state || response.state)) return false;
  return applySessionState(response.session_state || response.state, { force: true });
}

async function sendMessage(options = {}) {
  if (state.monitorReadOnly || state.queueSubmitting ||
      (state.pendingPrompt && state.pendingPrompt.sid === state.currentSessionId)) return false;
  // Send while the mic is armed means "finish dictating, then send": stop the
  // recording and let the final transcript land in the composer BEFORE the
  // Send/Queue/Steer/Stop decision below reads it — otherwise a mid-recording
  // click posts a half-transcribed draft (or hits the empty-composer Stop
  // path) while the mic keeps appending text afterwards. A second click while
  // the flush drains must not double-submit.
  if (state.voiceSendFlushing) return false;
  if (
    (state.audioState === "recording" || state.audioState === "processing") &&
    typeof stopRecording === "function"
  ) {
    const voiceSid = state.currentSessionId;
    state.voiceSendFlushing = true;
    updateSendState();
    try {
      if (state.audioState === "recording") {
        try {
          // stopRecording flips to "processing" synchronously, then drains:
          // final chunk, in-flight replies, whole-recording review → idle.
          const stop = stopRecording();
          if (stop && typeof stop.catch === "function") {
            stop.catch((err) => console.warn("voice flush failed:", err));
          }
        } catch (err) {
          console.warn("voice flush failed:", err);
        }
      }
      // Bound the wait: a hung transcription request must not deadlock Send
      // forever — but it must not release early either. Continuing while the
      // drain still runs posts a half transcript and lets late chunks land in
      // the cleared composer, or (empty draft + busy tree) hits the
      // cancelStreaming path. On timeout: bail, keep the draft, let the
      // drain finish into the intact composer; the next Send click re-enters
      // this wait and submits the completed transcript.
      const deadline = Date.now() + (Number(state.voiceFlushTimeoutMs) || 30000);
      while (state.audioState === "processing" && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 50));
      }
      if (state.audioState === "processing") {
        if (typeof appendSystemMessage === "function") {
          appendSystemMessage(
            "Still transcribing — press Send again when the text lands.", true,
          );
        }
        return false;
      }
    } finally {
      state.voiceSendFlushing = false;
      updateSendState();
    }
    if (state.currentSessionId !== voiceSid) return false;
  }
  // Uploads must settle before deciding Send / Queue / Steer / Stop. In
  // particular an attachment-only draft must not accidentally stop a turn.
  const pendingAttachments = [
    ...(state.pendingImageReads || []), ...(state.pendingTextReads || []),
  ];
  if (pendingAttachments.length) {
    const uploadSession = state.currentSessionId;
    await Promise.all(pendingAttachments);
    if (state.currentSessionId !== uploadSession || state.monitorReadOnly || state.queueSubmitting) return false;
  }
  // A staged or failed routing selection can never leak into a prompt. Save or
  // Discard must resolve the sheet first.
  ensureModelSheetState();
  if (["OPEN_DIRTY", "APPLYING", "OPEN_ERROR"].includes(state.modelSheet.phase)) {
    setModelSheetPhase(
      state.modelSheet.phase === "APPLYING" ? "APPLYING" : state.modelSheet.phase,
      state.modelSheet.error,
    );
    const sheet = document.getElementById("agent-model-pop");
    if (sheet) sheet.focus();
    return false;
  }
  // Read the trimmed input up-front so we can decide whether to enqueue or stop.
  const composerText = messageInput.value.trim();

  // Empty composer + open session: the primary button is Stop only while the
  // session or its subagent tree has active work (sessionHasActiveWork) —
  // even when the local busy flags lag the gateway (right after the parent
  // turn_ended, or in a monitor subagent view whose child liveness is unknown
  // until the next poll). A TRULY idle session must not fire a cancel at all:
  // the button shows plain Send there and an empty click is a no-op.
  if (
    state.currentSessionId &&
    !composerText &&
    !state.pastedImages.length &&
    !state.attachedTextFiles.length
  ) {
    const hasWork = typeof sessionHasActiveWork === "function"
      ? sessionHasActiveWork()
      : !!(state.isStreaming || state.treeLive);
    if (hasWork) cancelStreaming();
    return false;
  }

  // A parked question is a real mid-turn decision boundary, not ordinary
  // streaming. A typed draft means Steer: abort/unblock that ask, wait for its
  // terminal event, then submit the untouched draft as a fresh turn. The
  // adjacent tree-stop button remains the independent Abort action.
  if (
    !options.skipPendingAskSteer &&
    typeof composerWillSteerPendingAsk === "function" &&
    composerWillSteerPendingAsk()
  ) {
    return steerPendingAskFromComposer();
  }

  // Parent turn still open (isStreaming):
  //   - images → Steer (cancel + send ACP image blocks; /queue is text-only).
  //   - non-empty plain prompt / text-file attachments → enqueue.
  //   - empty / slash → Stop (cancel tree).
  // Parent done but subagents live (treeLive, !isStreaming):
  //   - images → Steer so the gateway cannot strip vision blocks.
  //   - empty → Stop tree; typed text falls through to a normal Send below.
  const hasImages = !!(state.pastedImages && state.pastedImages.length);
  if (
    hasImages &&
    !options.skipBusyQueue &&
    (state.isStreaming || state.treeLive) &&
    typeof steerBusyTurnWithDraft === "function"
  ) {
    return steerBusyTurnWithDraft();
  }
  // A steer (skipBusyQueue) has already cancelled the live turn and painted
  // its own prompt bubble. If the busy flags are still up because the abort's
  // turn_ended has not landed, it must still send a plain prompt: the queue
  // branch below would paint a second bubble and report "new turn".
  if (state.isStreaming && !state._askStopReleased && !options.skipBusyQueue) {
    if (composerWillQueue()) {
      const queueSid = state.currentSessionId;
      state.queueSubmitting = true;
      renderHarnessWidget();
      updateSendState();
      try {
        if (state.pendingImageReads && state.pendingImageReads.length) {
          await Promise.all(state.pendingImageReads.splice(0));
        }
        if (state.pendingTextReads && state.pendingTextReads.length) {
          await Promise.all(state.pendingTextReads.splice(0));
        }
        const textFiles = (state.attachedTextFiles || []).slice();
        const queueText = composerText;
        const cacheLenBefore = (state.msgCache || []).length;
        const outcome = await enqueueQueuedMessage(
          queueSid, queueText, composerText, _attachmentRefs(textFiles),
        );
        if (outcome === "queued" && state.currentSessionId === queueSid) {
          if (typeof clearAttachments === "function") clearAttachments();
          messageInput.value = "";
          messageInput.style.height = "auto";
          // Brief "queued" hint in the status bar.
          const prev = statusText.textContent;
          statusText.textContent = "queued ✓";
          setTimeout(() => {
            if (statusText.textContent === "queued ✓") statusText.textContent = prev;
          }, 2000);
          return true;
        }
        if (outcome === "new-turn" && state.currentSessionId === queueSid) {
          // The gateway answered `accepted` WITHOUT `queued`: it started a new
          // turn instead of appending a follow-up entry (the composer only
          // queues while the parent turn is open). The text IS delivered, so
          // clear it, paint the user prompt bubble so it is not lost from view,
          // say nothing more: the bubble and the new turn already show it.
          if (typeof clearAttachments === "function") clearAttachments();
          messageInput.value = "";
          messageInput.style.height = "auto";
          // The live stream may have delivered this turn's turn_started (and
          // painted the prompt) before the POST response arrived. Painting
          // again would duplicate the bubble, and forcing isStreaming would
          // leave the UI busy after a turn that already ended.
          const streamPainted = (state.msgCache || []).slice(cacheLenBefore)
            .some((m) => m && m.role === "user");
          if (!streamPainted) {
            const echoText = composerText;
            const chips = _attachmentChips(textFiles);
            const optEl = appendMessage(
              "user",
              echoText,
              true,
              chips.length ? chips : null,
              null,
              null,
              new Date().toISOString(),
              viewerAuthor(state.me),
            );
            state.pendingEcho = { sid: queueSid, text: queueText, echoText, images: 0, el: optEl };
            updateSendState();
          }
          return true;
        }
        if (outcome === "pending") return true;
        // Rejected: keep the draft so the operator can retry.
        return false;
      } catch (e) {
        if (state.currentSessionId === queueSid) {
          appendSystemMessage(`Queue error: ${escapeHtml(e.message)}`, true);
        }
        return false;
      } finally {
        state.queueSubmitting = false;
        renderHarnessWidget();
        updateSendState(); // draft gone → flip queue-button back to Stop
      }
    }
    cancelStreaming();
    return false;
  }
  if (state.treeLive && !composerHasDraft()) {
    cancelStreaming();
    return false;
  }

  // First-click lock: createSession / model-apply can take seconds, and the
  // composer still holds the draft until we snapshot it below. Extra Send /
  // Enter clicks in that window used to POST the same prompt N times.
  if (state.promptSubmitting) return false;
  state.promptSubmitting = true;
  updateSendState();
  if (typeof statusText !== "undefined" && statusText) {
    statusText.textContent = "sending…";
  }
  try {
  const restoreComposerDraft = (draftText, draftImages, draftFiles) => {
    if (draftText && !messageInput.value) messageInput.value = draftText;
    if (draftImages && draftImages.length) state.pastedImages = draftImages;
    if (draftFiles && draftFiles.length) state.attachedTextFiles = draftFiles;
    if ((draftImages && draftImages.length) || (draftFiles && draftFiles.length)) {
      renderPastePreview();
    }
  };
  let text = messageInput.value.trim();
  // Settle in-flight paste/picker reads first — paste→Enter must not race the
  // FileReader and silently drop the image or text file.
  if (state.pendingImageReads.length) {
    await Promise.all(state.pendingImageReads.splice(0));
  }
  if (state.pendingTextReads.length) {
    await Promise.all(state.pendingTextReads.splice(0));
  }
  const images = state.pastedImages.slice();
  const textFiles = state.attachedTextFiles.slice();
  if (!text && !images.length && !textFiles.length) return false;

  // Slash commands route to /api/command. `/goal <objective>` also falls
  // through to the same /prompt + EventSource path as a typed message.
  // The You-bubble keeps the typed `/goal …` line; the widget is the only
  // extra chrome — no separate "goal created" command card.
  if (text.startsWith("/")) {
    const goalPrompt = typeof slashGoalSetPrompt === "function"
      ? slashGoalSetPrompt(text)
      : "";
    const typedCommand = text;
    messageInput.value = "";
    messageInput.style.height = "auto";
    sendBtn.disabled = true;
    hideSlashMenu();
    // `/goal` (and other session-mutating commands) must run inside a real
    // gateway session. In a fresh chat there is no currentSessionId yet, so
    // create one FIRST — otherwise the command goes out session-less and the
    // backend rejects it ("session not found: ''"), so the goal never gets
    // created. The normal chat path below already guards this for prompts.
    if (commandRequiresSession(text) && !state.currentSessionId) {
      if (!(await createSession(state.draftTopic, { commit: true }))) {
        updateSendState();
        return false;
      }
    }
    const cmdResp = await runCommand(text, { silent: !!goalPrompt });
    const cmdFailed = !cmdResp || cmdResp.ok === false ||
      cmdResp.error || ((cmdResp.result || {}).kind === "error");
    if (!goalPrompt || cmdFailed) {
      if (goalPrompt && cmdFailed) {
        const detail =
          ((cmdResp || {}).result || {}).text ||
          (cmdResp && (cmdResp.error || cmdResp.message)) ||
          "goal command failed";
        appendSystemMessage(detail, true);
      }
      messageInput.focus();
      updateSendState(); // staged images (preview stays visible) keep send enabled
      return true;
    }
    text = typedCommand;
    if (goalPrompt) state.goalDismissedAfterMet = false;
  }

  // A completed goal keeps a ✓ badge until the next ordinary turn. That
  // send hides the widget so the conversation returns to a normal chat.
  // A new `/goal <objective>` (above) starts a fresh badge instead.
  const gw = document.getElementById("goal-widget");
  const widgetShowsMet = !!(gw && !gw.hidden && (
    gw.classList.contains("goal-met") ||
    !!gw.querySelector(".goal-met") ||
    /completed/i.test(gw.innerText || "")
  ));
  const isGoalSet = typeof slashGoalSetPrompt === "function" && !!slashGoalSetPrompt(text);
  if (
    !isGoalSet &&
    !state.goalDismissedAfterMet &&
    (String(state.goalTerminal || "").toLowerCase() === "met" || widgetShowsMet) &&
    typeof resetGoalTracker === "function"
  ) {
    state.goalDismissedAfterMet = true;
    resetGoalTracker();
  }

  // The prompt is what the operator typed. Saved files travel as references
  // (``files``); the backend adds the model-facing attachment context, so the
  // transcript shows the typed text plus thumbnails/chips.
  const fullText = options.queuePrompt ?? text;
  const echoText = text;
  const fileRefs = _attachmentRefs(textFiles, images);
  const fileChips = _attachmentChips(textFiles);

  // Clear the composer BEFORE createSession / model-apply. Those awaits are
  // the window where a second click used to resubmit the same draft. Restore
  // in catch (and on createSession failure) so a 413/network error doesn't
  // destroy the user's text or attachments.
  clearAttachments();
  messageInput.value = "";
  messageInput.style.height = "auto";
  sendBtn.disabled = true;

  // Create session if needed. If a navigation supersedes the create mid-flight,
  // abort the send so this turn can't land in a later-selected session.
  if (!state.currentSessionId) {
    const effectiveSelection = cloneModelSelection(state.modelSheet.effective);
    const turnAgent = effectiveSelection.agent || (typeof selectedPickerAgent === "function" ? selectedPickerAgent() : state.pickerAgent) || state.currentAgent || "jarvis";
    const turnModel = effectiveSelection.wire_id || effectiveSelection.alias ||
      (state.pickerModels || {})[turnAgent] || (typeof pickerModelAlias === "function" ? pickerModelAlias(turnAgent) : "") || "";
    const optModeEl = document.getElementById("opt-mode");
    const pickerMode = state.modePickerTouched ? ((optModeEl && optModeEl.value) || "") : "";

    const ok = await createSession(state.draftTopic, {
      commit: true,
      agent: turnAgent,
      model: turnModel,
      mode: pickerMode,
    });
    if (!ok) {
      restoreComposerDraft(text, images, textFiles);
      updateSendState();
      return false;
    }
  }
  const sid = state.currentSessionId;
  // Snapshot the composer-owned agent/model pair for THIS turn. Picker changes
  // are frontend-only; this send boundary is the first point where they are
  // applied to the backend session.
  const effectiveSelection = cloneModelSelection(state.modelSheet.effective);
  const turnAgent = effectiveSelection.agent || state.currentAgent;
  const turnModel = effectiveSelection.wire_id || effectiveSelection.alias ||
    (state.pickerModels || {})[turnAgent] || "";
  // A New Chat button click creates only a web-store placeholder. Agent/mode
  // picker changes on that empty row are pending intent, not a session
  // switch. Carry them on the first submit so the server mints the gateway
  // session for the selected agent and applies the selected mode before turn 1.
  // Reconcile against server authority first: a premature probe may have bound
  // the placeholder while the client still thinks it is fresh. Never skip
  // the agent update when the server is already bound to a different agent —
  // initial_agent cannot rebind.
  const initialIntent = firstIntentForSession(sid, turnAgent);
  const pendingSessionIntent = !!initialIntent.initial_agent;
  // Only turn 1 names its model, and only a catalogue-checked id (loaded now,
  // at Send, when the page has none yet).
  const initialModel = pendingSessionIntent && typeof firstTurnInitialModel === "function"
    ? await firstTurnInitialModel(turnAgent, turnModel) : "";
  const turnRows = state.modelOptionsByAgent[turnAgent] || [];
  const turnRow = turnRows.find((row) =>
    row.id === effectiveSelection.alias &&
    (!effectiveSelection.wire_id || row.wireId === effectiveSelection.wire_id)
  );
  // Turn 1 also names the effort shown in the sheet; the server applies it
  // to the new session and remembers it for the model.
  const initialEffort = pendingSessionIntent && turnRow && turnRow.reasoning &&
    effectiveSelection.reasoning != null
    ? String(effectiveSelection.reasoning) : "";

  // Apply the frontend model setting only before the first turn. Later model
  // changes are acknowledged by saveModelSheet(), so repeating /model here
  // would race an active turn and produce a turn_active rejection.
  const session = state.sessions && state.sessions[sid];
  const hasPriorTurn = !!(
    state.isStreaming ||
    state.treeLive ||
    (session && (
      session.active ||
      session.tree_busy ||
      session.state === "waiting" ||
      (Array.isArray(session.messages) && session.messages.length > 0)
    ))
  );
  const forceModel = pendingSessionIntent ? "" : modelToApplyBeforeTurn(
    turnModel, state.engineModel, hasPriorTurn,
  );
  const forceModelCommand = forceModel
    ? (modelApplyCommand(effectiveSelection, turnRow,
        effectiveSelection.reasoning) || `/model ${forceModel}`)
    : "";
  // Likewise make the web's reasoning/tools DEFAULTS authoritative for turn 1,
  // so the first message renders streamed 💭 + 🔧 even if the gateway session was
  // born (or adopted) with different visibility. Applied just before the prompt
  // POST, same as forceModel.
  const forceVis = pendingSessionIntent
    ? []
    : visToApplyBeforeTurn(state.vis, state.engineModel);

  // Durable display URLs are filled in after the artifact upload below; the
  // optimistic bubble is painted immediately with data: thumbs so a slow
  // model-apply does not look like a missed click. Fail-closed paths drop it.
  let displayImageUrls = images.length
    ? images.map((i) => i.url || i.dataUrl)
    : null;
  if (fileChips.length) displayImageUrls = [...(displayImageUrls || []), ...fileChips];
  let optimisticEchoEl = (options && options.optimisticEchoEl) || null;

  // Session-scoped so a switch can't mis-dedup; image count so turn_started's
  // echo of this same prompt (which now carries the image blocks) dedups too.
  state.pendingEcho = { sid, text: fullText, echoText, images: images.length, el: optimisticEchoEl };
  const submission = {
    sid, requestId: clientRequestId(), draft: text, echo: state.pendingEcho,
    accepted: false, images, files: textFiles,
  };
  state.pendingPrompt = submission;

  // A new operator prompt is not an abort. A leftover Stop (_instantAbort)
  // used to drop every later SSE frame except turn_ended, so the
  // reply only appeared after refresh rematerialized history.
  state._instantAbort = false;

  updateSendState(); // submitting is transport state; turn_started owns running
  _turnIndicatorStart("sending"); // live status: sending → received after POST
  renderStatusSegments();
  // Safety watchdog: if isStreaming stays true for 5 minutes without a
  // turn_ended, auto-clear it (events stream may have died mid-turn).
  _startStreamingWatchdog();

  // Reset auto-scroll so the user immediately follows their new message
  if (state.scroll) state.scroll.autoScroll = true;
  if (typeof VirtualScroller !== "undefined") {
    VirtualScroller._hideNewMsgBtn();
    VirtualScroller._forcePin();
  }

  // Paint the user bubble NOW — before agent-switch / upload / model-apply —
  // so the operator sees the click land instead of clicking Send again.
  if (!optimisticEchoEl) {
    optimisticEchoEl = appendMessage(
      "user",
      echoText,
      true,
      displayImageUrls,
      null,
      null,
      new Date().toISOString(),
      viewerAuthor(state.me),
      { preserveLiveTurn: true },
    );
  }
  if (state.pendingEcho) state.pendingEcho.el = optimisticEchoEl || null;

  // ── Submit + subscribe (spec §1, §7, §8) ────────────────────────────────
  // The interface no longer drives the turn: it SUBMITS the prompt (fire-and-
  // forget, 202 + start_offset) and SUBSCRIBES to the canonical event stream
  // from that offset. The user echo and the assistant reply both render FROM
  // the stream (turn_started → user bubble, text_delta/… → assistant bubble),
  // so there is NO local echo and no double render. The gateway owns the turn;
  // navigating away can't cancel it (the log is the source of truth).
  let accepted = false;
  try {
    // A bound session changes agent only when a turn is actually sent.
    // Merely opening/closing the picker never mutates backend state.
    if (!pendingSessionIntent && turnAgent !== state.currentAgent) {
      const switched = await applyAgentSwitch(turnAgent);
      if (!switched || state.currentAgent !== turnAgent)
        throw new Error(state.switchError || `Could not switch to ${turnAgent}`);
      if (state.currentSessionId !== sid) return false;
    }
    // Persist user-attached images into the authenticated artifact store so
    // thumbs open via GET /files/<sid>/<digest>/<name> (same jail as agent
    // outputs). Keep base64 ACP blocks for the model prompt either way.
    if (images.length) {
      const uploaded = [];
      for (const img of images) {
        let url = img.url || "";
        if (!url && img.dataUrl) {
          try {
            const m = /^data:([^;,]+);base64,(.*)$/.exec(img.dataUrl) || [];
            const mime = m[1] || "image/png";
            const ext =
              mime === "image/jpeg" || mime === "image/jpg"
                ? ".jpg"
                : mime === "image/gif"
                  ? ".gif"
                  : mime === "image/webp"
                    ? ".webp"
                    : ".png";
            let name = String(img.name || "attached-image").replace(
              /[^\w.\-]+/g,
              "_",
            );
            if (!/\.(png|jpe?g|gif|webp)$/i.test(name)) name += ext;
            const art = await apiPost(
              `/api/sessions/${encodeURIComponent(sid)}/artifacts`,
              {
                name,
                data_base64: img.dataUrl,
                content_addressed: true,
              },
            );
            if (art && art.url) url = art.url;
          } catch (uploadErr) {
            // Best-effort: still send the vision prompt with base64; the thumb
            // falls back to the data URL inside the lightbox.
            console.warn("user image artifact upload failed", uploadErr);
          }
        }
        if (url) img.url = url;
        uploaded.push(url || img.dataUrl);
      }
      displayImageUrls = uploaded;
    }

    // Apply the picker's visible model to the gateway session BEFORE the prompt
    // (same `/model` path a manual picker change uses), awaited so the session
    // adopts it before the turn reads its model.
    // Logical rejections (model_not_found, …) stay fail-closed. Transport /
    // closed-channel failures (AcpTransportClosed → HTTP 500) are non-fatal for
    // image attaches so a best-effort model command must not block the vision
    // prompt. Fail-closed paths drop the optimistic bubble in catch.
    if (forceModelCommand) {
      try {
        await runCommand(forceModelCommand, {
          silent: true,
          throwOnError: true,
          firstIntent: initialIntent,
        });
      } catch (cause) {
        const detail = switchFailureMessage(cause);
        const code = String((cause && cause.code) || "");
        const msg = String((cause && cause.message) || detail || "");
        const transportish =
          /^http[_\s-]?\d+/i.test(code) ||
          /acptransportclosed|gateway unreachable|network|econn|fetch failed|502|503|500/i.test(
            `${code} ${msg}`,
          );
        // Image attaches treat transport/closed model-apply failure as a
        // warning so the user can still deliver the vision prompt on the live
        // engine. Text-only sends stay fail-closed.
        const mayContinue = transportish && images.length > 0;
        if (!mayContinue) {
          const error = new Error(
            `Could not apply model ${forceModel}; prompt was not sent: ${detail}`,
          );
          error.code = "model_switch_failed";
          error.data = { code: error.code, message: error.message };
          throw error;
        }
        appendSystemMessage(
          `Warning: could not apply model ${escapeHtml(String(forceModel || ""))} (${escapeHtml(detail)}); sending on the current session model.`,
          true,
        );
      }
      if (state.currentSessionId !== sid) return false; // switched away while applying
    }
    // Assert the web's reasoning/tools defaults (stream + on) to the gateway
    // session before the turn reads them — symmetric with the model apply above.
    // Best-effort + per-command session-guarded so a switch mid-apply can't leak
    // visibility into the wrong session.
    for (const cmd of forceVis) {
      try {
        await runCommand(cmd, {
          silent: true,
          firstIntent: initialIntent,
        });
      } catch (_) {
        /* fall through: run on the session default */
      }
      if (state.currentSessionId !== sid) return; // switched away while applying
    }

    // With images the prompt is sent as an ACP content-blocks LIST — text
    // blocks plus {type:'image', data, mimeType}, the exact shape the
    // Telegram bot uses. The bridge's _blocks_to_user_content only converts
    // THIS shape for the vision model (unknown types like raw image_url are
    // silently dropped), and the converted parts become turn_started's
    // `user` field, so the images survive reloads in every interface.
    // When a durable /files URL was uploaded, also stamp `url` so history
    // parsers prefer it over reconstructing a multi-MB data URL.
    const payload = images.length
      ? {
          prompt: [
            ...(fullText ? [{ type: "text", text: fullText }] : []),
            ...images.map((img) => {
              const m = /^data:([^;,]+);base64,(.*)$/.exec(img.dataUrl) || [];
              const block = {
                type: "image",
                data: m[2] || "",
                mimeType: m[1] || "image/png",
              };
              if (img.url) block.url = img.url;
              return block;
            }),
          ],
          expected_revision: state.sessionRevision || undefined,
          ...initialIntent,
          ...(pendingSessionIntent && initialModel ? { initial_model: initialModel } : {}),
          ...(initialEffort ? { initial_effort: initialEffort } : {}),
        }
      : {
          prompt: fullText,
          expected_revision: state.sessionRevision || undefined,
          ...initialIntent,
          ...(pendingSessionIntent && initialModel ? { initial_model: initialModel } : {}),
          ...(initialEffort ? { initial_effort: initialEffort } : {}),
        };
    if (fileRefs.length) payload.files = fileRefs;
    payload.request_id = submission.requestId;
    const promptPath = `/api/sessions/${encodeURIComponent(sid)}/prompt`;
    const rejectSubmission = (error) => {
      if (submission.accepted || state.currentSessionId !== sid) return;
      if (state.pendingPrompt === submission) state.pendingPrompt = null;
      if (state.pendingEcho === submission.echo) state.pendingEcho = null;
      dropOptimisticUserEcho(optimisticEchoEl);
      optimisticEchoEl = null;
      restoreComposerDraft(text, images, textFiles);
      appendSystemMessage(`Error: ${escapeHtml(switchFailureMessage(error))}`, true);
      if (!state.isStreaming) _turnIndicatorStop();
      statusText.textContent = "Error";
      updateSendState();
    };
    const resp = await apiPost(promptPath, payload);
    if (resp && resp.pending) {
      if (!submission.accepted && state.currentSessionId === sid) {
        restoreComposerDraft(text, images, textFiles);
        statusText.textContent = "Submission pending…";
        watchPromptReceipt(submission, promptPath, payload, () => {}, rejectSubmission);
      }
      if (state.currentSessionId === sid && _liveStreamNeedsOpen(sid)) openLiveStream(sid, 0);
      return true; // command is pending; never treat it as rejected/restore a steer
    }
    if (!resp || resp.accepted !== true) throw new Error((resp && resp.error) || "Prompt rejected");
    acceptPromptReceipt(submission);
    accepted = true;
    if (state.currentSessionId !== sid) return true; // switched away mid-submit
    // Gateway accepted the prompt. Sequence is sending → received →
    // warming (this POST), then SSE advances prefill/reasoning/writing.
    // Do not wait for the first stream frame to leave "prompt received" —
    // a remount or quiet engine left the composer stuck on that string.
    if (_turn && (_turn.phase === "sending" || _turn.phase === "received")) {
      _turnIndicatorMark("received");
      _turnIndicatorTick();
      _turnIndicatorMark("warming");
    }
    applyFirstSubmitSession(resp);
    const startOffset =
      resp && typeof resp.start_offset === "number" ? resp.start_offset : 0;
    // Subscribe from this turn's first JSONL byte. from= is a cursor into
    // the same file the EventSource tails — not a CLI vs ACP choice.
    // A page that already tails this conversation must keep that
    // EventSource: close+reopen from start_offset drops the healthy
    // connection and the reply never paints (✓ prompt received + Stop).
    if (pendingSessionIntent || _liveStreamNeedsOpen(sid)) {
      openLiveStream(sid, startOffset);
    }
    // Keep the "thinking…" status until turn_ended — don't clear isStreaming
    // here. The events stream's turn_ended handler does that.
  } catch (e) {
    // A canonical receipt can beat even a failed/legacy proxy response.
    if (submission.accepted) return true;
    if (state.currentSessionId !== sid) return false;
    if (state.pendingPrompt === submission) state.pendingPrompt = null;
    state.pendingEcho = null; // failed send — don't suppress a later turn_started
    dropOptimisticUserEcho(optimisticEchoEl);
    optimisticEchoEl = null;
    // A rejected/stale prompt was never accepted by the canonical log. Restore
    // the draft and refresh authority so retry cannot land on the wrong lane.
    restoreComposerDraft(text, images, textFiles);
    if (e && e.code === "stale_revision") {
      try {
        await refreshSessionState(sid, state.navEpoch);
      } catch (_) {}
    }
    appendSystemMessage(`Error: ${escapeHtml(switchFailureMessage(e))}`, true);
    statusText.textContent = "Error";
    if (!state.isStreaming) _turnIndicatorStop(); // a rejection cannot stop another live turn
  } finally {
    sendBtn.disabled = false;
    // Refresh the sidebar only if the user has it open (the rest list is a
    // lazy load on menu expand; the active row updates via the state frame).
    if (typeof _menuSectionOpen === "function" && _menuSectionOpen("sidebar-conversations")) {
      loadSessionList();
      loadTopicList(); // a new turn may have created/renamed a topic's latest
    }
    messageInput.focus();
    if (!state.isStreaming) updateSendButton();
  }
  return accepted;
  } finally {
    state.promptSubmitting = false;
    if (typeof updateSendState === "function") updateSendState();
    if (!state.isStreaming && statusText && statusText.textContent === "sending…") {
      statusText.textContent = "";
    }
  }
}

// Legacy /api/chat SSE event handler — retained for the (kept) /api/chat
// fallback path; the inverted live turn renders via renderCanonicalEvent.
function handleStreamEvent(event, messageEl, contentEl) {
  switch (event.type) {
    case "delta": {
      // Accumulate against the element's stashed raw text (NOT the rendered
      // DOM, whose textContent would include reasoning-block text).
      const raw = (messageEl.dataset.raw || "") + event.content;
      setAssistantContent(messageEl, raw);
      break;
    }

    case "error":
      contentEl.innerHTML = `<span style="color:var(--error)">Error: ${escapeHtml(event.error)}</span>`;
      statusText.textContent = "Error";
      break;

    case "message_end":
      // Final content — ensure it's rendered (collapses the reasoning block).
      if (event.content) {
        setAssistantContent(messageEl, event.content);
      }
      // The gateway tags the terminal frame with the engine's REAL model
      // (e.g. claude-opus-4-…/gpt-5.5-codex/composer-2.5) for CLI/ACP
      // agents, and authoritatively echoes "" when it has cleared it (the
      // selection changed). Sync the badge in lock-step — SET on a concrete
      // id, CLEAR on "" — whenever the key is present, so it never lingers
      // on a stale model. Distinct from the agent-roster picker.
      if ("model" in event) {
        state.engineModel = event.model || "";
        // The gateway tags the terminal frame with the
        // operator-facing alias alongside the raw id; absent alias → the
        // raw id is the display label.
        state.engineModelAlias = event.model_alias || "";
        updateHeader();
      }
      // A turn can change teacher supervision out-of-band — the model (or
      // teacher) may set/clear a goal, or a proactive teacher fire flips the
      // gate. Re-sync the indicator from the gateway so a goal-driven
      // transition shows without a manual reload (the toggle path already
      // updates live via its state_patch).
      syncTeacherState();
      statusText.textContent = "";
      break;
  }
}

function abortStreamingInstant() {
  const abortedPid = String(state.livePhysicalTurnId || "");
  if (abortedPid && (state.isStreaming || state.liveTurn)) {
    if (!state.locallyAbortedTurnIds) state.locallyAbortedTurnIds = {};
    state.locallyAbortedTurnIds[abortedPid] = true;
  }
  state._instantAbort = true;
  _stopStreaming({ ignoreStaleActivity: true });
  if (state.liveTurn) {
    state.liveTurn.bs.abortFast("");
    renderBlockTurn(state.liveTurn);
    state.liveTurn = null;
    state.pendingEcho = null;
    statusText.textContent = "";
  }
  // A parked question must settle even if its terminal frame arrives after
  // the next turn starts (when it is correctly ignored as a stale closer).
  if (typeof _finalizePendingAskUserCards === "function") {
    _finalizePendingAskUserCards("⏹ Question aborted");
  }
}

// Cancel the in-flight turn (spec §9.3). With the inverted submit+subscribe
// model the turn runs server-side, so cancel targets the gateway turn via the
// cancel route — the confirmation surfaces as turn_ended(aborted) on the live
// stream (renderCanonicalEvent finalizes the bubble). Falls back to aborting a
// local /api/chat fetch if one is somehow active.
//
// HARD cancel (force=true) — the web Stop button is an /abort, not a graceful
// /stop: it must halt generation immediately (SIGKILL + killpg the worker, flip
// the engine's _cancelled flag, break the round/dispatch loops now). A graceful
// force=false lets the in-flight streaming round FINISH, so a long answer keeps
// generating after Stop — not what the button implies.
function cancelStreaming(options = {}) {
  const sid = state.currentSessionId;
  const previous = state.pendingCancel;
  if (previous && previous.sid === sid && previous.promise) return previous.promise;
  const pending = previous && previous.sid === sid ? previous : {
    sid, requestId: clientRequestId(), promise: null,
  };
  state.pendingCancel = pending;
  let cancelRequest = Promise.resolve({ accepted: false });
  if (sid) {
    const hash = typeof sessionGwHash === "function"
      ? sessionGwHash(sid)
      : sid;
    // A delegated child page stops ONE subagent (non-cascading) via the
    // lineage-authorized abort endpoint — the same endpoint the monitor row
    // ✕ uses — which writes turn_ended(aborted) on the child and delivers a
    // terminal subagent_result to the parent. The plain session-cancel route
    // reaches the CLI-engine shim, which does not fold the child's own row
    // out of the live set. Fall back to the cancel route when the
    // abort proxy refuses (e.g. the ancestor is not gateway-owned). The parent
    // id rides the chat stream's ``state.live`` (stored on the row by
    // ``_applyLiveSnapshotToSession``) — no /live GET. Do NOT fetch the global
    // agent tree just to decide this.
    cancelRequest = (async () => {
      const parentId = (state.sessions && state.sessions[sid]
        && state.sessions[sid].parent_session_id) || "";
      if (hash && parentId) {
        try {
          const ok = await abortSubagentSession(
            hash, { force: true, silent: true },
          );
          if (ok) return { accepted: true };
        } catch (_e) { /* fall through to the cancel route */ }
      }
      // force=true cascades to child sessions + background subtree (gateway).
      return apiPost(`/api/sessions/${encodeURIComponent(sid)}/cancel`, {
        force: true, request_id: pending.requestId,
      });
    })().catch((error) => {
      if (options.awaitServer) throw error;
      return null;
    }).then((res) => {
      // Reuse the id only while the outcome is unknown; after a verdict a
      // new Stop press is a new command.
      if (res && !res.pending && state.pendingCancel === pending) pending.requestId = clientRequestId();
      return res;
    }).finally(() => {
      pending.promise = null;
      if (state.currentSessionId === sid) updateSendState();
    });
    pending.promise = cancelRequest;
  }
  if (state.abortController) {
    try {
      state.abortController.abort();
    } catch (_e) {}
  }
  updateSendState();
  if (options.awaitServer) return cancelRequest;
  cancelRequest.catch(() => {});
}

// ── Streaming UI helpers ────────────────────────────────────────────────

/** Drive the primary composer button through its states:
 *    idle                         → Send
 *    pending ask + draft          → Steer (dedicated Abort remains visible)
 *    isStreaming + draft          → Queue (tree-stop-btn also shown)
 *    isStreaming + empty          → Stop tree
 *    treeLive (!streaming) + empty→ Stop tree
 *    treeLive + draft             → Send new message to jarvis (tree-stop-btn shown)
 */
function updateSendButton() {
  if (!sendBtn) return;
  const sendIcon = sendBtn.querySelector(".send-icon");
  const stopIcon = sendBtn.querySelector(".stop-icon");
  const steeringAsk = composerWillSteerPendingAsk() ||
    (typeof composerWillSteerBusy === "function" && composerWillSteerBusy());
  const queueing = !steeringAsk && composerWillQueue();
  // Stop on the main button only while the session (or its subagent tree)
  // actually has active work: parent turn open (isStreaming), any live
  // descendant / waiting parent (treeLive), or the session parked on owner
  // input (paused/ask — sessionNeedsStop). A fully idle session shows the
  // plain Send state: there is nothing to stop. treeBusy() covers the
  // streaming + tree cases; the web stays a pure renderer, so staleness is
  // bounded by the 3s tree poll, not by removing the gate.
  const stopping =
    !state.monitorReadOnly && !!state.currentSessionId &&
    !queueing && !composerHasDraft() && sessionHasActiveWork();
  sendBtn.classList.toggle("stop-btn", stopping);
  sendBtn.classList.toggle("queue-btn", queueing);
  sendBtn.classList.toggle("steer-btn", steeringAsk);
  if (stopping) {
    sendBtn.title = sessionHasPendingAsk()
      ? "Abort pending question"
      : state.treeLive && !state.isStreaming
        ? "Stop session tree (subagents still running)"
        : "Stop generation";
    sendBtn.setAttribute("aria-label", sendBtn.title);
    sendBtn.disabled = !!state.monitorReadOnly || !!state.voiceSendFlushing;
    if (sendIcon) sendIcon.style.display = "none";
    if (stopIcon) stopIcon.style.display = "";
  } else {
    sendBtn.title = composerWillSteerPendingAsk()
      ? "Steer — abort the pending question and send this message"
      : (typeof composerWillSteerBusy === "function" && composerWillSteerBusy())
        ? "Steer — stop this turn and send the draft with attachments"
      : queueing
        ? "Queue follow-up — delivered to jarvis after this turn"
        : "Send message";
    sendBtn.setAttribute(
      "aria-label",
      steeringAsk ? "Steer turn" : queueing ? "Queue follow-up" : "Send message",
    );
    if (stopIcon) stopIcon.style.display = "none";
    if (sendIcon) sendIcon.style.display = "";
  }
  syncTreeStopBtn();
}

/** Dedicated Stop while the main button is Send/Queue/Steer (draft present) so the
 *  operator can kill the subagent tree without clearing the draft. Shown only
 *  while the session or its subagent tree has active work — an idle session
 *  with a draft has nothing to kill, so it stays a plain Send. */
function syncTreeStopBtn() {
  const btn = document.getElementById("tree-stop-btn");
  if (!btn) return;
  const show =
    !state.monitorReadOnly && !!state.currentSessionId &&
    composerHasDraft() && sessionHasActiveWork();
  btn.hidden = !show;
  btn.disabled = !!state.monitorReadOnly;
  btn.title = sessionHasPendingAsk()
    ? "Abort pending question"
    : "Stop session tree (parent + subagents)";
  btn.setAttribute("aria-label", btn.title);
}

// States where the OPEN session itself is parked on the operator even though
// nothing is streaming and the tree is idle: an explicit pause (pause_resume
// lifecycle), or the backend's derived ``ask`` (session parked on a pending
// ask_user/tool-permission approval — its tree node still shows the turn's
// live state, which the poll keeps flagged busy anyway; ``ask`` is defensive
// for servers that emit it). Stop stays available so the operator can abort
// out of the pause. (``error``/``closed`` are terminal — nothing to stop.)
const _NEEDS_INPUT_STATES = new Set(["paused", "ask"]);

// Durable states where the open session has no own turn left to stop. A
// leftover activity.own_live / can_stop bit on the row must not keep Stop
// armed here; only a live descendant tree or a parked question can.
// ``waiting`` belongs here too: it has no own turn either (live_turn_idx is
// NULL), and a wedged row — last child's clear event never landed, or a
// hidden recap side-call on an older backend — stays ``waiting`` with zero
// live descendants/jobs, where Stop could kill nothing anyway.
const _IDLE_SESSION_STATES = new Set(["idle", "error", "closed", "succeeded", "waiting"]);

/** True when the open session is parked on owner input (paused / ask). */
function sessionNeedsStop() {
  const s = state.sessions && state.sessions[state.currentSessionId];
  if (!s) return false;
  const a = s.activity && typeof s.activity === "object" ? s.activity : null;
  if (a && ["question", "permission", "operator_paused"].includes(a.attention))
    return true;
  return _NEEDS_INPUT_STATES.has(String(s.state || ""));
}

/** True when an AskUserQuestion card is on screen and still awaiting a pick. */
function sessionHasPendingAsk() {
  const cards = state.askUserCards;
  let pending = false;
  let sawCard = false;
  if (cards) {
    for (const id of Object.keys(cards)) {
      sawCard = true;
      const c = cards[id];
      if (c && !c.answered && !c.finalized) pending = true;
    }
  }
  if (pending) return true;
  // A settled local card (Submit or Stop) is the source of truth. The
  // session can stay ``ask`` until the next poll; that must not keep the
  // Chat composer on Stop or treat the next draft as Steer.
  if (sawCard) return false;
  return sessionNeedsStop();
}

/** True when the draft should steer a parked question instead of queueing. */
function composerWillSteerPendingAsk() {
  return !!(
    !state.monitorReadOnly &&
    state.currentSessionId &&
    composerHasDraft() &&
    sessionHasPendingAsk()
  );
}

/** Cancel a live turn/tree and submit the current draft, including images. */
async function steerBusyTurnWithDraft() {
  if (state.monitorReadOnly || state.promptSubmitting) return false;
  const sid = state.currentSessionId;
  if (!sid) return false;
  state.promptSubmitting = true;
  updateSendState();
  if (statusText) statusText.textContent = "steering…";
  try {
    await cancelStreaming({ awaitServer: true });
    await waitForCancelSettlement();
    if (state.currentSessionId !== sid) {
      throw new Error("the session changed before the steering message was sent");
    }
    state.promptSubmitting = false;
    updateSendState();
    return await sendMessage({ skipPendingAskSteer: true, skipBusyQueue: true });
  } catch (error) {
    state._instantAbort = false;
    if (state.currentSessionId === sid) {
      appendSystemMessage(
        `Steer error: ${escapeHtml(error && error.message || "cancel failed")}`,
        true,
      );
    }
    return false;
  } finally {
    state.promptSubmitting = false;
    if (state.currentSessionId === sid) updateSendState();
  }
}

/** Abort a parked ask and submit the existing composer draft as the next turn. */
async function steerPendingAskFromComposer() {
  if (!composerWillSteerPendingAsk() || state.promptSubmitting) return false;
  const sid = state.currentSessionId;
  state.promptSubmitting = true;
  updateSendState();
  if (statusText) statusText.textContent = "steering…";
  try {
    await cancelStreaming({ awaitServer: true });
    await waitForCancelSettlement();
    if (state.currentSessionId !== sid) {
      throw new Error("the session changed before the steering message was sent");
    }
    // Release the cancel/steer lock before entering the ordinary send path;
    // sendMessage owns its own first-click lock and preserves/restores drafts.
    state.promptSubmitting = false;
    updateSendState();
    return await sendMessage({ skipPendingAskSteer: true });
  } catch (error) {
    // The draft never left the composer. A failed cancel must not leave the
    // local instant-abort gate swallowing later live events.
    state._instantAbort = false;
    if (state.currentSessionId === sid) {
      appendSystemMessage(
        `Steer error: ${escapeHtml(error && error.message || "cancel failed")}`,
        true,
      );
    }
    return false;
  } finally {
    state.promptSubmitting = false;
    if (state.currentSessionId === sid) updateSendState();
  }
}

/** Apply a live snapshot onto the open session row.
 *  The single chat stream's ``state.live`` frame is the only producer now (the
 *  chat page no longer polls ``GET /api/sessions/<sid>/live``). Keeps
 *  ``activity`` / ``capabilities`` / ``tree_busy`` / ``parent_session_id`` in
 *  sync with ``state.treeLive`` so sticky-Stop cannot linger on a stale
 *  can_stop bit, and so the cancel path can read the parent id locally. */
function _applyLiveSnapshotToSession(sid, data) {
  if (!sid || !data || !state.sessions || !state.sessions[sid]) return;
  if (_isStaleLiveSnapshot(sid, data)) return;
  const row = state.sessions[sid];
  row._liveSnapshot = data;
  row.state = (data.session_state || row.state || "idle");
  row.live = !!data.tree_busy;
  row.tree_busy = !!data.tree_busy;
  row.own_live = !!data.own_live;
  for (const key of [
    "activity", "version", "phase", "attention", "lifecycle",
    "last_outcome", "own_live", "stale", "active_descendant_count",
    "open_job_count", "phase_changed_at", "heartbeat_at", "outcome_at",
    "status", "display", "capabilities",
  ]) {
    if (Object.prototype.hasOwnProperty.call(data, key)) row[key] = data[key];
  }
  // Parent + child list ride the same snapshot; the cancel path reads
  // ``row.parent_session_id`` instead of a per-action /live GET.
  if (Object.prototype.hasOwnProperty.call(data, "parent_session_id")) {
    row.parent_session_id = String(data.parent_session_id || "");
  }
  if (Array.isArray(data.subagents)) row.subagents = data.subagents;
  // When the live payload nests activity, prefer that as the authoritative
  // sticky-Stop source.
  if (data.activity && typeof data.activity === "object") {
    row.activity = data.activity;
  } else if (!data.tree_busy) {
    // Clear a stale can_stop when the cheap bit says the tree is idle.
    if (row.activity && typeof row.activity === "object") {
      row.activity = Object.assign({}, row.activity, {
        tree_busy: false,
        own_live: false,
        capabilities: Object.assign(
          {},
          (row.activity.capabilities && typeof row.activity.capabilities === "object")
            ? row.activity.capabilities : {},
          { can_stop: false, can_send: true },
        ),
      });
    }
    if (row.capabilities && typeof row.capabilities === "object") {
      row.capabilities = Object.assign({}, row.capabilities, {
        can_stop: false,
        can_send: true,
      });
    }
  }
}

/** True when there is something to stop: the session itself is busy (turn
 *  open / streaming), any subagent in the tree is busy (treeLive), or the
 *  session is waiting on owner input. Fully idle → false → plain Send state. */
function sessionHasActiveWork() {
  if (
    state._askStopReleased
    && typeof askUserHasPendingCard === "function"
    && !askUserHasPendingCard()
  ) {
    return false;
  }
  if (!state.currentSessionId) return false;
  const s = state.sessions && state.sessions[state.currentSessionId];
  const a = s && s.activity && typeof s.activity === "object" ? s.activity : null;
  const st = String((s && (s.state || s.session_state)) || "");
  // A durable idle/terminal session has no own turn left to stop. The next
  // ``state`` frame can be rejected as stale or suppressed as an unchanged
  // ``hb``, so a leftover activity.own_live / can_stop / optimistic treeLive
  // must not keep the red Stop visible next to a plain Send. Only a genuinely
  // live descendant tree (the server's counts) or a parked question still has
  // something to abort.
  if (_IDLE_SESSION_STATES.has(st)) {
    const snap = typeof _currentLiveSnapshot === "function" ? _currentLiveSnapshot() : null;
    const liveTree = !!(snap && (
      snap.descendants_running === true
      || Number(snap.active_descendant_count) > 0
      || Number(snap.open_job_count) > 0
    ));
    return !!(state.isStreaming || sessionHasPendingAsk() || liveTree);
  }
  // Live client flags first. A stale activity.can_stop / tree_busy left on
  // the session row (SSE updates treeLive without always rewriting activity)
  // must NOT keep the red Stop visible after the UI already shows idle.
  const busy = typeof treeBusy === "function"
    ? treeBusy()
    : !!(state.isStreaming || state.treeLive);
  if (busy || sessionHasPendingAsk()) return true;
  if (typeof sessionNeedsStop === "function" && sessionNeedsStop()) return true;
  if (a && a.own_live) return true;
  if (a && (a.tree_busy || (a.capabilities && a.capabilities.can_stop))) return true;
  return false;
}

// ── Per-subagent stop (non-cascading) ────────────────────────────────────
// Stops ONE delegated child session via the gateway abort endpoint. The web
// proxies identity/lineage (never browser query params); the gateway claims +
// kills the child's runner, SIGKILLs its durable engine-pidfile subtree on a
// hard press, writes turn_ended(aborted) on the child, and delivers a terminal
// subagent_result to the parent (which wakes to see the delegation cancelled).
// Grandchildren and siblings keep running.
async function abortSubagentSession(sid, opts = {}) {
  if (!sid) return false;
  try {
    await apiPost("/api/agents/abort", {
      session_id: sid,
      force: !!(opts && opts.force) || undefined,
    });
  } catch (error) {
    // ``silent`` (the Stop-button fallback path) suppresses the banner: a
    // lineage-gate refusal there just falls through to the tree-cancel
    // route, so a scary "Abort error" toast would be misleading.
    if (!(opts && opts.silent) && typeof appendSystemMessage === "function") {
      appendSystemMessage(`Abort error: ${escapeHtml(error.message)}`, true);
    }
    return false;
  }
  if (typeof refreshTreeLive === "function") refreshTreeLive();
  return true;
}

function _hkey(h) {
  return String(h || "").slice(0, 24);
}

function _findTreeNode(tree, hash) {
  const want = _hkey(hash);
  if (!want || !tree) return null;
  let found = null;
  function walk(n) {
    if (_hkey(n.hash) === want) { found = n; return true; }
    for (const c of n.children || []) if (walk(c)) return true;
    return false;
  }
  for (const a of tree.agents || []) {
    for (const r of a.roots || []) if (walk(r)) break;
    if (found) break;
  }
  return found;
}

/** Parent hash of the node whose key matches ``hash`` — "" for a root or
 *  when the node is absent. Mirrors ``_findTreeNode``'s key normalization.
 *  Lets the chat page tell "this open session is a delegated subagent"
 *  (then its Stop goes through the per-subagent abort endpoint, mirroring
 *  the monitor row ✕) from "this is a root/tree stop" (plain cancel). */
function _treeParentHash(tree, hash) {
  if (!tree || !hash) return "";
  const want = _hkey(hash);
  function walk(n, parentHash) {
    if (_hkey(n.hash) === want) return parentHash || "";
    for (const c of n.children || []) {
      const found = walk(c, n.hash || "");
      if (found !== null) return found;
    }
    return null;
  }
  for (const a of tree.agents || []) {
    for (const r of a.roots || []) {
      const found = walk(r, "");
      if (found !== null) return found;
    }
  }
  return "";
}

// States that mean a session node is STILL doing work — mirrors the gateway's
// ``_CHILD_BUSY_STATES`` (session_index.py): prompting/prefill/reasoning/
// streaming/tool_running/compacting/ask/paused/waiting. Anything else
// (idle/error/closed) is at rest: nothing to stop.
const _TREE_BUSY_STATES = new Set([
  "prompting", "prefill", "reasoning", "streaming", "tool_running",
  "compacting", "paused", "waiting", "ask",
]);

/** True when ONE tree node has active work. ``live`` is the tree's freshness
 *  gate (state register touched recently), so an abandoned mid-turn node ages
 *  out instead of keeping Stop forever, while a freshly-``idle`` node stops
 *  being busy immediately. ``waiting`` is special-cased: it is a live
 *  orchestrator state even when the freshness ``live`` flag briefly lags a
 *  child's fold (and the gateway sweeps abandoned live states to idle on
 *  boot, so an unconditional ``waiting`` cannot stick forever). */
function _nodeBusy(n) {
  if (!n) return false;
  const a = n.activity && typeof n.activity === "object" ? n.activity : null;
  if (a && a.tree_busy) return true;
  const st = String(n.state || "");
  if (st === "waiting") return true;
  return !!n.live && _TREE_BUSY_STATES.has(st);
}

function _nodeOrDescendantLive(n) {
  if (!n) return false;
  if (_nodeBusy(n)) return true;
  for (const c of n.children || []) {
    if (_nodeOrDescendantLive(c)) return true;
  }
  return false;
}

function _liveBits(data) {
  const a = (data && data.activity && typeof data.activity === "object")
    ? data.activity : {};
  const src = data || {};
  return {
    session_state: String(src.session_state || src.state || a.phase || ""),
    tree_busy: !!(src.tree_busy || a.tree_busy),
    own_live: !!(src.own_live || a.own_live),
  };
}

function _ownTurnLiveFrom(data) {
  const b = _liveBits(data);
  return !!(b.own_live || _parentTurnIsLive(b.session_state));
}

function _isStaleLiveSnapshot(sid, data) {
  const row = state.sessions && state.sessions[sid];
  const boundary = Number(row && row._lastTurnBoundarySeq) || 0;
  return boundary > 0 && (Number(data && data.snapshot_seq) || 0) < boundary;
}

function _currentLiveSnapshot() {
  const sid = state.currentSessionId;
  const row = sid && state.sessions && state.sessions[sid];
  // Keep the delivered live frame intact: sidebar/header merges must not
  // combine an old count or waiting phase with a newer snapshot's fields.
  const snap = row && (row._liveSnapshot || row);
  if (!snap || _isStaleLiveSnapshot(sid, snap)) {
    return { session_state: "idle", tree_busy: false, own_live: false,
      descendants_running: false };
  }
  return snap;
}

function _phaseFromSessionState(st) {
  const s = String(st || "");
  if (s === "tool_running") return "tool";
  if (typeof _PHASE !== "undefined" && _PHASE[s]) return s;
  return "warming";
}

/** Own turn owns the ticker + top bar. "waiting on subagents…" is only for
 *  a finished parent whose descendants are still live. tree_busy includes
 *  own tool_running — that must NOT paint the waiting label or hide the bar. */
function _shouldIgnoreStaleOwnLive() {
  return !!(state._ignoreStaleActivity && !state.liveTurn && !state.promptSubmitting);
}

function _paintComposerStatus() {
  if (typeof updateAgentMonAccordion === "function") updateAgentMonAccordion();
  if (typeof statusText === "undefined" || !statusText) return;
  const snap = _currentLiveSnapshot();
  if (_ownTurnLiveFrom(snap) && !_shouldIgnoreStaleOwnLive()) {
    _ensureOwnTurnUI(snap);
    return;
  }
  if (state.isStreaming || _turn) return;
  if (snap.descendants_running === true) {
    if (statusText.textContent !== "waiting on subagents…") {
      statusText.textContent = "waiting on subagents…";
    }
    return;
  }
  if (statusText.textContent === "waiting on subagents…") statusText.textContent = "";
}

function _ensureOwnTurnUI(data) {
  if (_shouldIgnoreStaleOwnLive()) return;
  const bits = _liveBits(data);
  const phase = _phaseFromSessionState(bits.session_state);
  if (_turn && state.isStreaming) {
    if (phase && phase !== _turn.phase) {
      _turnIndicatorMark(phase, 0, phase === "tool" ? (_turn.tool || "tool") : undefined);
    }
    return;
  }
  if (typeof _applyActiveTurnUI === "function") {
    _applyActiveTurnUI({ phase: phase });
  }
}

function _setTreeLive(live) {
  const next = !!live;
  if (state.treeLive === next) {
    _paintComposerStatus();
    return;
  }
  state.treeLive = next;
  updateSendState();
  if (typeof renderStatusSegments === "function") renderStatusSegments();
  if (typeof updateAgentMonAccordion === "function") updateAgentMonAccordion();
  _paintComposerStatus();
}

const _PARENT_TURN_LIVE = {
  prompting: 1, prefill: 1, reasoning: 1, streaming: 1, tool_running: 1,
  compacting: 1, paused: 1, ask: 1, retrying: 1,
};

function _parentTurnIsLive(sessionState) {
  return !!_PARENT_TURN_LIVE[String(sessionState || "")];
}

/** Align the ticker / Stop / top bar with durable activity.
 *  Own-live or a live parent phase STARTS the running UI (join / reload /
 *  missed turn_started). Only a finished own turn may stop it. */
function _reconcileStreamingFromActivity(data) {
  if (_isStaleLiveSnapshot(state.currentSessionId, data)) return;
  const bits = _liveBits(data);
  if (data && !bits.tree_busy && !bits.own_live &&
      ["idle", "aborted", "error", "closed", "succeeded"].includes(bits.session_state)) {
    state.pendingCancel = null;
    state._instantAbort = false;
    state._askStopReleased = false;
    state.steeringQueueId = "";
    state.treeLive = false;
    _finalizePendingAskUserCards("Turn ended");
    // An in-flight prompt has no turn yet: idle is not its verdict, so keep
    // the submission status until its receipt or turn_started settles it.
    const submitting = state.promptSubmitting || !!(state.pendingPrompt
      && state.pendingPrompt.sid === state.currentSessionId);
    if (submitting && !state.isStreaming) {
      updateSendState();
      return;
    }
    _stopStreaming({ ignoreStaleActivity: true });
    return;
  }
  if (_ownTurnLiveFrom(data)) {
    if (_shouldIgnoreStaleOwnLive()) {
      _paintComposerStatus();
      return;
    }
    _ensureOwnTurnUI(data);
    return;
  }
  const sessState = String((data && data.session_state) || "idle");
  if (typeof sessionHasPendingAsk === "function" && sessionHasPendingAsk()) return;
  if (!state.isStreaming && !_turn) {
    _paintComposerStatus();
    return;
  }
  // ``waiting`` means descendants are live, not that this tab's in-flight
  // follow-up ended. A just-sent prompt still reads waiting/idle until
  // turn_started projects prompting — snapping that race paints ● idle and
  // kills the elapsed ticker. turn_ended already calls _stopStreaming.
  if (sessState === "waiting") {
    _paintComposerStatus();
    return;
  }
  if (state.promptSubmitting && (sessState === "idle" || sessState === "waiting"))
    return;
  if (typeof _stopStreaming === "function") _stopStreaming();
}

/** Reconcile ``state.treeLive`` from the live/activity snapshot already
 *  delivered on the single chat stream (``state.live`` → the open session
 *  row). The stream pushes a fresh ``state`` frame on every turn boundary (see
 *  ``iter_session_stream``'s terminal wake), so there is no per-session
 *  ``/live`` REST poll. While a live source is open, ``_applyState`` has
 *  already reconciled — nothing to do; with none open, fall back to the last
 *  snapshot the stream applied. */
async function refreshTreeLive() {
  if (typeof uiSseIsOpen === "function" && uiSseIsOpen()) return;
  if (!state.currentSessionId || state.monitorReadOnly) {
    _setTreeLive(false);
    return;
  }
  const snap = _currentLiveSnapshot();
  if (typeof renderSessionList === "function") renderSessionList();
  _setTreeLive(!!snap.tree_busy);
  _reconcileStreamingFromActivity(snap);
}

/**
 * True when the open tab has no usable EventSource for ``sid``.
 * Covers brand-new chats AND the mid-reconnect / dead-stream gap where
 * ``liveSid`` still matches but ``liveSource`` is null or CLOSED — without
 * this, a follow-up prompt never reopens the stream and the UI goes dark.
 */
function _liveStreamNeedsOpen(sid) {
  if (!sid) return true;
  if (state.liveSid !== sid) return true;
  const es = state.liveSource;
  if (!es) return true;
  try {
    if (typeof EventSource !== "undefined" && es.readyState === EventSource.CLOSED) {
      return true;
    }
  } catch (_e) {
    return true;
  }
  return false;
}

/**
 * Safety watchdog: auto-clear isStreaming if the stream goes silent and
 * turn_ended never arrives. This is an INACTIVITY timer — every live gateway
 * frame (and turn_started) re-arms it, so it only fires after 5 minutes with NO
 * contact from the gateway. A long but actively-streaming turn (a cursor-agent
 * CLI doing minutes of real work) must NOT trip it; only a truly dead stream
 * should.
 */
function _startStreamingWatchdog() {
  _stopStreamingWatchdog();
  state._streamingWatchdog = setTimeout(() => {
    if (state.isStreaming) {
      if (typeof sessionHasPendingAsk === "function" && sessionHasPendingAsk()) {
        _startStreamingWatchdog();
        return;
      }
      console.warn("Streaming watchdog: stream silent 5min — auto-clearing");
      _stopStreaming({ ignoreStaleActivity: true });
    }
  }, 300000); // 5 minutes of SILENCE (re-armed on every gateway frame)
}
function _stopStreamingWatchdog() {
  if (state._streamingWatchdog) {
    clearTimeout(state._streamingWatchdog);
    state._streamingWatchdog = null;
  }
}

/** Clean up streaming state — called from turn_ended, error, and watchdog. */
function _stopStreaming(opts) {
  state.isStreaming = false;
  if (opts && opts.ignoreStaleActivity) state._ignoreStaleActivity = true;
  _stopStreamingWatchdog();
  _turnIndicatorStop();
  // turn_ended is authoritative for THIS tab's turn. A stale /live snapshot
  // can still say tool_running after the card closed — ignoreStaleActivity
  // keeps _ensureOwnTurnUI / _reconcile from relighting the ticker. A later
  // turn_started or join (_turnIndicatorStart) clears the latch.
  // "waiting on subagents…" only when descendants remain.
  const snap = _currentLiveSnapshot();
  const waiting = snap.descendants_running === true;
  _setTreeLive(waiting);
  // turn_ended is authoritative for THIS tab's turn, so drop the row's
  // own-live projection here instead of waiting for the next ``state`` frame.
  // That frame can be rejected as stale (``_isStaleLiveSnapshot``) or
  // suppressed as an unchanged ``hb``; either way a leftover
  // ``activity.own_live`` / ``can_stop`` would keep the red Stop armed next to
  // a plain Send long after the session went idle.
  const row = state.sessions && state.sessions[state.currentSessionId];
  if (row) {
    if (row.activity && typeof row.activity === "object") {
      const next = Object.assign({}, row.activity, { own_live: false });
      if (!waiting) {
        next.tree_busy = false;
        next.capabilities = Object.assign(
          {},
          (row.activity.capabilities && typeof row.activity.capabilities === "object")
            ? row.activity.capabilities : {},
          { can_stop: false },
        );
      }
      row.activity = next;
    }
  }
  if (typeof statusText !== "undefined" && statusText) {
    statusText.textContent = waiting ? "waiting on subagents…" : "";
  }
  renderStatusSegments();
  updateSendState(); // recomputes disabled + icon (Send/Stop) in one place
}

// ── Live "model is running" indicator ──────────────────────────────────────
// A static "thinking…" reads as frozen during prefill / a long tool / non-
// streamed reasoning. This drives an ALWAYS-MOVING ticker (phase · tokens ·
// elapsed) + a top progress bar from the canonical event stream, and surfaces a
// calm "waiting on the model" note when events go quiet — so a slow turn never
// looks like an error (you shouldn't feel the urge to cancel a working turn).
const _SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const _PHASE = {
  sending: { icon: "·", label: "sending" },
  received: { icon: "✓", label: "prompt received" },
  prompting: { icon: "·", label: "warming up" },
  warming: { icon: "·", label: "warming up" },
  prefill: { icon: "·", label: "prefill" },
  reasoning: { icon: "·", label: "reasoning" },
  writing: { icon: "·", label: "writing" },
  streaming: { icon: "·", label: "writing" },
  tool: { icon: "·", label: "running" },
  compacting: { icon: "·", label: "compacting context" },
  ask: { icon: "·", label: "needs input" },
  retrying: { icon: "·", label: "retrying" },
};
let _turn = null;

// A viewer that JOINS an already-running turn (monitor deep-link, second tab)
// has no turn_started in its tail window — light up the "working" UI from the
// REST snapshot's `active` flag so status/stop-button match the originating tab.
function _applyActiveTurnUI(opts) {
  if (typeof _shouldIgnoreStaleOwnLive === "function" && _shouldIgnoreStaleOwnLive()) {
    return;
  }
  const phase = opts && opts.phase;
  if (state.isStreaming && _turn) {
    if (phase && _PHASE[phase] && phase !== _turn.phase) {
      _turnIndicatorMark(phase, 0, phase === "tool" ? (_turn.tool || "tool") : undefined);
    }
    return;
  }
  state.isStreaming = true;
  _turnIndicatorStart(_PHASE[phase] ? phase : undefined);
  if (phase === "compacting") {
    _turnIndicatorMark("compacting", 0, "compact_context");
  } else if (phase === "retrying") {
    _turnIndicatorMark("retrying", 0);
  } else if (phase === "tool") {
    _turnIndicatorMark("tool", 0, "tool");
  }
  updateSendState();          // flip send → stop AND re-enable (renderMessages may
                              // have disabled the empty composer before streaming)
  _startStreamingWatchdog();  // auto-clear if the stream goes silent
  refreshTreeLive();
}

function _turnIndicatorStart(phase) {
  state._ignoreStaleActivity = false;
  if (_turn && _turn.timer) clearInterval(_turn.timer);
  _turn = {
    startAt: Date.now(),
    lastAt: Date.now(),
    phase: _PHASE[phase] ? phase : "warming",
    chars: 0,
    tool: "",
    toolAt: Date.now(),
    spin: 0,
    timer: null,
  };
  const bar = document.getElementById("turn-progress");
  if (bar) {
    bar.hidden = false;
    bar.classList.add("running");
  }
  _turn.timer = setInterval(_turnIndicatorTick, 250);
  _turnIndicatorTick();
  renderStatusSegments();
}
// The running indicator shows the active tool/command name. Some agents (e.g.
// the cursor ACP bridge) report the FULL shell command as the tool name, which
// can be hundreds of chars and blows the status line across the whole viewport.
// Clamp it to a single readable token with a middle ellipsis so both the head
// (the program) and tail (the target) stay legible. Pure + exported-by-name so
// the web test can pin the truncation contract.
function _truncCmd(s, max) {
  s = String(s == null ? "" : s).replace(/\s+/g, " ").trim();
  const lim = max || 80;
  if (s.length <= lim) return s;
  const head = Math.ceil((lim - 1) / 2);
  const tail = Math.floor((lim - 1) / 2);
  return s.slice(0, head) + "…" + s.slice(s.length - tail);
}
// Record activity: advance the phase, count streamed chars, refresh the clock.
function _turnIndicatorMark(phase, addChars, tool) {
  if (!_turn) return;
  const prev = _turn.phase;
  _turn.lastAt = Date.now();
  if (phase) _turn.phase = phase;
  if (addChars) _turn.chars += addChars;
  if (phase === "tool") {
    _turn.tool = _truncCmd(tool || "tool", 80);
    _turn.toolAt = Date.now();
  }
  if (phase && phase !== prev && typeof updateAgentMonAccordion === "function") {
    updateAgentMonAccordion();
  }
}
function _fmtDur(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  return s < 60
    ? s + "s"
    : Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
}
function _turnIndicatorTick() {
  if (!_turn || !statusText) return;
  _turn.spin = (_turn.spin + 1) % _SPIN.length;
  const ph = _PHASE[_turn.phase] || _PHASE.warming;
  let txt;
  if (_turn.phase === "received") {
    // Confirmation only — no widget, no duration/token suffix.
    txt = `${ph.icon} ${ph.label}`;
  } else if (_turn.phase === "tool") {
    txt = `${_SPIN[_turn.spin]} ${ph.icon} ${ph.label} ${_turn.tool} · ${_fmtDur(Date.now() - _turn.toolAt)}`;
  } else if (_turn.phase === "retrying" && _turn.tool) {
    txt = `${_SPIN[_turn.spin]} ${ph.icon} ${ph.label} ${_turn.tool} · ${_fmtDur(Date.now() - _turn.startAt)}`;
  } else {
    const tok = _turn.chars ? ` · ~${Math.round(_turn.chars / 4)} tok` : "";
    txt = `${_SPIN[_turn.spin]} ${ph.icon} ${ph.label}${tok} · ${_fmtDur(Date.now() - _turn.startAt)}`;
  }
  // Quiet-gap reassurance: a pause (prefill on big context, a slow tool) is
  // expected — say so rather than looking stuck. Skip on "received" so the
  // confirmation string stays exactly "✓ prompt received".
  if (_turn.phase !== "received" && Date.now() - _turn.lastAt > 9000)
    txt += " · still working, waiting on the model";
  statusText.textContent = txt;
}
function _turnIndicatorStop() {
  if (_turn && _turn.timer) clearInterval(_turn.timer);
  _turn = null;
  const bar = document.getElementById("turn-progress");
  if (bar) {
    bar.hidden = true;
    bar.classList.remove("running");
  }
}

// ── Audio / voice input ─────────────────────────────────────────────────

/** Toggle audio recording on/off. */

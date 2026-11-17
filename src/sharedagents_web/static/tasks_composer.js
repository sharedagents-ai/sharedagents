/* Tasks describe uses the Dev/Chat composer adapter and Chat's prompt API. */
(function () {
  "use strict";
  const composer = window.describeComposer;
  const modelButton = document.getElementById("tasks-agent-model");
  const topicInput = document.getElementById("tasks-describe-topic");
  const chatLink = document.getElementById("tasks-created-chat");
  const picker = new AgentModelDialog({
    allowDefault: true,
    onSave: choice => {
      selection = choice;
      renderSelection();
    },
  });
  let selection = null;
  let created = null;
  let submission = null;
  let named = false;
  let busy = false;
  let selectedTopic = "";

  function renderSelection() {
    modelButton.textContent = selection.agent + " · " + (selection.label || selection.model || "Default model");
  }
  const ready = picker.picker.load().then(() => {
    selection = { agent: picker.picker.getAgent(), model: picker.picker.getModelId(),
                  label: picker.picker.getModel()?.label };
    renderSelection();
    modelButton.disabled = false;
  });
  modelButton.addEventListener("click", () => picker.open(selection));

  window.describeSubmit = async function () {
    if (busy) return;
    busy = true;
    state.promptSubmitting = true;
    try {
      if (state.audioState === "recording") await stopRecording();
      if (state.audioState === "processing") return;
      await composer.waitPending();
      if (!composer.hasDraft()) return;
      if (composer.text().trim().length > 12000) throw new Error("Describe the task in up to 12000 characters.");
      composer.setFrozen(true);
      modelButton.disabled = true;
      topicInput.disabled = true;
      await ready;
      modelButton.disabled = true;
      if (!created) {
        appendSystemMessage("Creating task…");
        selectedTopic = topicInput.value;
        created = await apiPost("/api/tasks", { agent: selection.agent, model: selection.model,
          ...(selectedTopic ? { topic: selectedTopic } : {}) });
        chatLink.href = BASE + "/?session=" + encodeURIComponent(created.session.id);
        chatLink.hidden = false;
        document.dispatchEvent(new CustomEvent("sharedagents:tasks-refresh"));
      }
      const text = composer.text().trim();
      if (!named) {
        selectedTopic = topicInput.value;
        appendSystemMessage("Creating title and topic…");
        await apiPost("/api/tasks/" + encodeURIComponent(created.id) + "/describe", {
          description: text || "Review attached files: " + _attachmentRefs(state.attachedTextFiles, state.pastedImages).map(f => f.name).join(", ").slice(0, 1000),
          ...(selectedTopic ? { topic: selectedTopic } : {}),
        });
        named = true;
        document.dispatchEvent(new CustomEvent("sharedagents:tasks-refresh"));
      }
      if (!submission) {
        const images = state.pastedImages.map(img => {
          const match = /^data:([^;,]+);base64,(.*)$/.exec(img.dataUrl || "");
          return { type: "image", data: match?.[2] || "", mimeType: match?.[1] || "image/png" };
        });
        submission = {
          prompt: images.length ? [...(text ? [{ type: "text", text }] : []), ...images] : text,
          files: _attachmentRefs(state.attachedTextFiles, state.pastedImages),
          initial_model: selection.model || undefined,
          request_id: crypto.randomUUID ? crypto.randomUUID() :
            Array.from(crypto.getRandomValues(new Uint8Array(16)), v => v.toString(16).padStart(2, "0")).join(""),
        };
      }
      appendSystemMessage("Starting task…");
      const result = await apiPost("/api/sessions/" + encodeURIComponent(created.session.id) + "/prompt", submission);
      if (!result.accepted) throw new Error("The task was not accepted. Retry Send or open the created task.");
      window.location.assign(chatLink.href);
    } catch (error) {
      appendSystemMessage(error.message || "Could not create task. Retry Send.");
    } finally {
      busy = false;
      state.promptSubmitting = false;
      // Retain the created session and exact request on retry, including an
      // ambiguous network failure. Never create a second task or prompt id.
      composer.setFrozen(!!created);
      modelButton.disabled = !!created;
      topicInput.disabled = !!created && named;
      if (created) {
        sendBtn.disabled = false;
        sendBtn.title = "Retry starting task";
      } else {
        initMic();
      }
    }
  };

  const host = document.getElementById("tasks-composer");
  const form = document.getElementById("tasks-describe-form");
  new ResizeObserver(() => {
    document.getElementById("tasks-app").style.setProperty("--tasks-composer-height", form.offsetHeight + "px");
  }).observe(form);
  host.addEventListener("input", () => {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + "px";
  });
  if (window.visualViewport) {
    const lift = () => form.style.setProperty("--tasks-keyboard-inset",
      Math.max(0, innerHeight - visualViewport.height - visualViewport.offsetTop) + "px");
    visualViewport.addEventListener("resize", lift);
    visualViewport.addEventListener("scroll", lift);
  }
})();

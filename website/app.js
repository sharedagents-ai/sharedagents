"use strict";

const views = {
  welcome: [
    "welcome-desktop.png",
    "Start with a conversation.",
    "Choose an agent and model, write your request, and continue in a persistent session.",
    "A new SharedAgents conversation, ready for a prompt and model selection.",
  ],
  tasks: [
    "tasks-desktop.png",
    "See what needs your attention.",
    "Follow sample tasks through Needs input, Working, and Completed, with their agent and topic in view.",
    "The SharedAgents task board with three sample documentation tasks.",
  ],
  activity: [
    "activity-desktop.png",
    "Keep usage in view.",
    "Switch between providers, models, tokens, costs, and request counts. All figures shown here are demo data.",
    "The SharedAgents usage chart with synthetic local and hosted model activity.",
  ],
  development: [
    "development-desktop.png",
    "Understand the delivery process.",
    "The development view explains the phases from a clear request through verification and delivery.",
    "The SharedAgents development view explaining its delivery phases.",
  ],
};

const image = document.querySelector("#tour-image");
const dialog = document.querySelector("#image-dialog");
document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    const view = views[button.dataset.view];
    if (!view) return;
    image.src = "assets/demo/" + view[0];
    image.alt = view[3];
    document.querySelector("#tour-title").textContent = view[1];
    document.querySelector("#tour-description").textContent = view[2];
    document.querySelectorAll("[data-view]").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
  });
});
document.querySelector("#copy-install").addEventListener("click", async () => {
  const status = document.querySelector("#copy-status");
  try {
    await navigator.clipboard.writeText(
      document.querySelector("#install-command").textContent,
    );
    status.textContent = "Installation command copied.";
  } catch {
    status.textContent = "Select and copy the installation command above.";
  }
});
document.querySelector("#expand-image").addEventListener("click", () => {
  const full = document.querySelector("#dialog-image");
  full.src = image.src;
  full.alt = image.alt;
  dialog.showModal();
});
document
  .querySelector("#close-image")
  .addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});

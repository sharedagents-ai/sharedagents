// Author attribution for user bubbles in a shared, multi-user session.
// Pure helpers (no DOM, no `state`) so tests load this file in Node.
// The label is the same in every browser and is never "You".

const AUTHOR_FALLBACK_LABEL = "Operator";
const AUTHOR_SOURCE_LABELS = Object.freeze({
  loop_supervisor: "Loop",
  goal_supervisor: "Goal",
  background_task: "Background task",
  steer: "Agent",
});

function authorDisplayName(actor) {
  if (!actor || typeof actor !== "object") return "";
  const name = String(actor.name || "").trim();
  if (name) return name;
  const email = String(actor.email || "").trim();
  return (email.split("@")[0] || "").trim() || email;
}

function normalizeAuthorActors(actors) {
  let list = [];
  if (Array.isArray(actors)) list = actors;
  else if (actors && typeof actors === "object") list = [actors];
  const seen = new Set();
  const out = [];
  for (const actor of list) {
    if (!actor || typeof actor !== "object") continue;
    const email = String(actor.email || "").trim().toLowerCase();
    const name = authorDisplayName(actor);
    const key = email || name;
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push({ email, name: name || email });
  }
  return out;
}

// author = {actors: [...], source: ""}; either key may be missing.
function authorInfo(author) {
  const a = author && typeof author === "object" ? author : {};
  const actors = normalizeAuthorActors(a.actors);
  let label = actors.map((x) => x.name).filter(Boolean).join(" & ");
  if (!label) {
    const source = String(a.source || "");
    label = Object.prototype.hasOwnProperty.call(AUTHOR_SOURCE_LABELS, source)
      ? AUTHOR_SOURCE_LABELS[source] : AUTHOR_FALLBACK_LABEL;
  }
  const emails = actors.map((x) => x.email).filter(Boolean);
  const first = Array.from(label.trim())[0] || "?";
  return {
    label,
    initial: first.toUpperCase(),
    emails,
    title: emails.length ? emails.join(", ") : label,
  };
}

// History message, turn_started event, or queue event → author descriptor.
function authorFromMessage(msg) {
  const m = msg && typeof msg === "object" ? msg : {};
  let actors = [];
  if (Array.isArray(m.actors)) actors = m.actors;
  else if (m.actor && typeof m.actor === "object") actors = [m.actor];
  return { actors, source: String(m.source || "") };
}

// The signed-in viewer ({email, name, via} from state.me) as an author.
function viewerAuthor(me) {
  return { actors: me && me.email ? [me] : [], source: "" };
}

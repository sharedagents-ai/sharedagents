"use strict";
(() => {
  const base = document.querySelector('meta[name="sharedagents-base"]').content,
    $ = (id) => document.getElementById(id);
  const workspace = new URLSearchParams(location.search).get("workspace");
  let selected = null,
    objects = [];
  const status = (text) => {
    $("ontology-status").textContent = text;
  };
  async function api(args, body) {
    const params = new URLSearchParams(args || {});
    if (workspace) params.set("workspace", workspace);
    const response = await fetch(
      base + "/api/ontology?" + params,
      body
        ? {
            method: "POST",
            credentials: "same-origin",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }
        : { credentials: "same-origin" },
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.error || "Ontology request failed");
    return result.value;
  }
  function svg(tag, attrs) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
    return el;
  }
  function render() {
    $("ontology-list").replaceChildren();
    objects.forEach((o) => {
      const button = document.createElement("button");
      button.className = "ontology-object";
      button.textContent = o.data.name;
      const meta = document.createElement("small");
      meta.textContent = o.kind + " · " + o.certification.status;
      button.append(meta);
      button.addEventListener("click", () => choose(o.id));
      $("ontology-list").append(button);
    });
    $("ontology-summary").textContent = objects.length + " objects shown";
    const canvas = $("ontology-graph");
    canvas.replaceChildren();
    const nodes = objects
      .filter((o) => ["term", "table", "metric"].includes(o.kind))
      .slice(0, 40);
    const positions = new Map();
    ["table", "term", "metric"].forEach((kind, lane) => {
      const group = nodes.filter((o) => o.kind === kind);
      group.forEach((o, i) =>
        positions.set(o.id, {
          x: 40 + lane * 255,
          y: 30 + ((i + 0.5) / group.length) * 430,
        }),
      );
    });
    const links = objects
      .filter((o) => o.kind === "relationship")
      .map((o) => [o.data.source, o.data.target]);
    nodes.forEach((o) =>
      (o.data.tables || []).forEach((t) => links.push([o.id, t])),
    );
    links.forEach(([a, b]) => {
      const p = positions.get(a),
        q = positions.get(b);
      if (p && q)
        canvas.append(svg("line", { x1: p.x, y1: p.y, x2: q.x, y2: q.y }));
    });
    nodes.forEach((o) => {
      const p = positions.get(o.id),
        circle = svg("circle", {
          cx: p.x,
          cy: p.y,
          r: 7,
          tabindex: 0,
          role: "button",
          "aria-label": o.data.name,
        }),
        text = svg("text", { x: p.x + 11, y: p.y + 4 });
      circle.addEventListener("click", () => choose(o.id));
      circle.addEventListener("keydown", (e) => {
        if (e.key === "Enter") choose(o.id);
      });
      text.textContent = o.data.name.slice(0, 30);
      canvas.append(circle, text);
    });
  }
  async function choose(id) {
    try {
      selected = await api({ action: "get", id });
      $("ontology-detail").hidden = false;
      $("ontology-name").textContent = selected.data.name;
      $("ontology-certification").textContent =
        selected.certification.status + " · " + selected.kind;
      $("ontology-edit-name").value = selected.data.name;
      $("ontology-description").value = selected.data.description || "";
      $("ontology-aliases").value = (selected.data.aliases || []).join(", ");
      $("ontology-data").textContent = JSON.stringify(selected, null, 2);
      const history = await api({ action: "history", id });
      $("ontology-history").replaceChildren();
      history.forEach((row) => {
        const li = document.createElement("li");
        li.textContent =
          row.created_at + " · " + row.operation + " · " + row.actor;
        $("ontology-history").append(li);
      });
    } catch (error) {
      status(error.message);
    }
  }
  async function load(query) {
    status("Loading ontology…");
    try {
      const value = await api(
        query
          ? { action: "search", q: query, limit: 200 }
          : { action: "list", limit: 200 },
      );
      objects = query ? value.map((row) => row.object) : value;
      render();
      status(
        objects.length
          ? "Select an object to inspect its meaning and provenance."
          : "No definitions yet. Start a research conversation to build the ontology.",
      );
    } catch (error) {
      status(error.message);
    }
  }
  $("ontology-search").addEventListener("submit", (e) => {
    e.preventDefault();
    load($("ontology-query").value);
  });
  $("ontology-refresh").addEventListener("click", () => {
    $("ontology-query").value = "";
    load();
  });
  $("ontology-edit").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selected) return;
    try {
      const data = {
        ...selected.data,
        name: $("ontology-edit-name").value,
        description: $("ontology-description").value,
        aliases: $("ontology-aliases")
          .value.split(",")
          .map((x) => x.trim())
          .filter(Boolean),
      };
      const saved = await api(
        {},
        {
          action: "edit",
          id: selected.id,
          kind: selected.kind,
          expected_revision: selected.revision,
          data,
        },
      );
      await load($("ontology-query").value);
      await choose(saved.id);
      status("Saved as a draft. The prior revision remains in history.");
    } catch (error) {
      status(error.message);
    }
  });
  async function review(certified) {
    if (!selected) return;
    try {
      const saved = await api(
        {},
        {
          action: "review",
          id: selected.id,
          expected_revision: selected.revision,
          certified,
          reason: $("ontology-reason").value,
        },
      );
      await load($("ontology-query").value);
      await choose(saved.id);
      status(
        certified ? "This revision is certified." : "This revision is a draft.",
      );
    } catch (error) {
      status(error.message);
    }
  }
  $("ontology-certify").addEventListener("click", () => review(true));
  $("ontology-unreview").addEventListener("click", () => review(false));
  $("ontology-export").addEventListener("click", async () => {
    try {
      const value = await api({ action: "export" });
      const url = URL.createObjectURL(
          new Blob([JSON.stringify(value, null, 2)], {
            type: "application/json",
          }),
        ),
        link = document.createElement("a");
      link.href = url;
      link.download = "ontology.json";
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      status(error.message);
    }
  });
  const role = document.body.dataset.ontologyRole;
  $("ontology-edit").hidden = role === "viewer";
  $("ontology-review").hidden = !["reviewer", "owner"].includes(role);
  load();
})();

/* Widget adaptation dictionary — keep in lockstep with
 * sharedagents_agent.tools.widget_adapt.WIDGET_ALIASES (see tests/test_widget_adapt.py).
 * Frontends must not hard-code Cursor / Claude / OpenCode tool titles.
 * The ask-user family especially: call `isQuestion()` (or `kind()`) instead of
 * listing tool names, or a foreign spelling loses its card and its place in the
 * turn (see the placement contract in ask_user.js).
 * `var` so Node/eval tests (BLOCKSTREAM_CORE extract) can bind it without `window`.
 */
var WidgetAdapt = {
  aliases: {
    todowrite: "todo",
    todo: "todo",
    todos: "todo",
    updatetodos: "todo",
    todolist: "todo",
    updateplan: "todo",
    taskcreate: "task",
    taskupdate: "task",
    taskstop: "task",
    tasklist: "task",
    taskget: "task",
    exitplanmode: "plan",
    createplan: "plan",
    create_plan: "plan",
    question: "question",
    askuserquestion: "question",
    userquestion: "question",
    askuser: "question",
  },
  key(name) {
    return String(name || "")
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");
  },
  kind(name) {
    return this.aliases[this.key(name)] || "";
  },
  isTodo(name) {
    return this.kind(name) === "todo";
  },
  isTask(name) {
    return this.kind(name) === "task";
  },
  isQuestion(name) {
    return this.kind(name) === "question";
  },
  adaptTodos(args) {
    const obj =
      args && typeof args === "object"
        ? args
        : typeof args === "string"
          ? (function () {
              try {
                return JSON.parse(args) || {};
              } catch (_e) {
                return {};
              }
            })()
          : {};
    const raw = Array.isArray(obj.todos)
      ? obj.todos
      : Array.isArray(obj.items)
        ? obj.items
        : Array.isArray(obj.plan)
          ? obj.plan
          : null;
    if (!raw) return null;
    const todos = raw
      .map((t) => {
        if (typeof t === "string") {
          const content = t.trim();
          return content
            ? { id: "", content, status: "pending", activeForm: "" }
            : null;
        }
        if (!t || typeof t !== "object") return null;
        const content = String(t.content || t.text || t.body || t.step || "").trim();
        if (!content) return null;
        let status = String(
          t.status ||
            t.state ||
            (t.completed === true
              ? "completed"
              : t.completed === false
                ? "pending"
                : "pending"),
        ).toLowerCase();
        if (status === "done") status = "completed";
        if (
          status !== "pending" &&
          status !== "in_progress" &&
          status !== "completed" &&
          status !== "cancelled"
        ) {
          status = "pending";
        }
        return {
          id: String(t.id || t.todoId || t.todo_id || ""),
          content,
          status,
          activeForm: String(t.activeForm || t.active_form || ""),
        };
      })
      .filter(Boolean);
    return todos.length ? { todos, merge: !!obj.merge } : null;
  },
};
if (typeof window !== "undefined") window.WidgetAdapt = WidgetAdapt;

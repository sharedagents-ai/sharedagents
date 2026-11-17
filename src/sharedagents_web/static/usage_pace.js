/* Pace row for Usage → Subscriptions: one normalized used % per plan. */
(function (root) {
  "use strict";

  var SCALE = 100;
  var WARN = 85;
  var SUB_IDS = ["codex", "claude", "cursor", "agy", "zcode", "mimo", "opencode", "devin"];
  var LEGEND =
    "Pace = used % ÷ share of the period already passed, for every limit the agent can spend " +
    "(5-hour, weekly, monthly, each model pool): how much of that limit is used by its reset at this rate. " +
    "The plan shows the highest pace among its pools with room left; limits are never averaged. " +
    "100% = on track to use it all exactly at reset · " +
    "under 100% = quota will expire unused, use more · over 100% = runs out before reset. " +
    "A used-up pool reads Limit hit on its own row until it resets; the plan reads Limit hit only when every pool is used up. " +
    "Green up to 85% · yellow up to 100% · red over 100%. Use next = lowest pace.";

  function isSubscriptionCard(card) {
    if (!card) return false;
    if (card.kind === "balance") return false;
    if (card.kind === "subscription") return true;
    return SUB_IDS.indexOf(String(card.id || "")) >= 0;
  }

  function scaleMax() {
    return SCALE;
  }

  function usedOf(card) {
    if (!card || card.pace_used_percent == null) return NaN;
    var value = Number(card.pace_used_percent);
    return Number.isFinite(value) ? value : NaN;
  }

  function formatPercent(value) {
    var n = Number(value);
    if (value == null || !Number.isFinite(n)) return "—";
    return Math.round(n) + "%";
  }

  function isRankable(card) {
    return isSubscriptionCard(card) &&
      card.pace_status === "ok" &&
      !card.pace_limit_hit &&
      !card.pace_engine_limit_hit &&
      Number.isFinite(usedOf(card));
  }

  function resetEpoch(card) {
    var t = Date.parse((card && card.pace_resets_at) || "");
    return Number.isFinite(t) ? t : Infinity;
  }

  // Lowest pace first; a tie goes to the plan whose limit resets sooner.
  function rankKey(card) {
    return [Math.round(usedOf(card) * 10) / 10, resetEpoch(card)];
  }

  function compareKeys(left, right) {
    var len = Math.max(left.length, right.length);
    for (var i = 0; i < len; i += 1) {
      if (left[i] < right[i]) return -1;
      if (left[i] > right[i]) return 1;
    }
    return 0;
  }

  function markUseNext(cards) {
    var list = cards || [];
    list.forEach(function (card) {
      if (card) card.pace_use_next = false;
    });
    var eligible = list.filter(isRankable);
    if (!eligible.length) return list;
    eligible.sort(function (a, b) {
      return compareKeys(rankKey(a), rankKey(b)) ||
        String(a.id || "").localeCompare(String(b.id || ""));
    });
    eligible[0].pace_use_next = true;
    return list;
  }

  function displaySortKey(card, index) {
    if (!isSubscriptionCard(card)) return [1, 0, 0, 0, index];
    if (isRankable(card)) return [0, 0].concat(rankKey(card), [index]);
    var group = { ok: 1, exhausted: 1, unknown: 2, not_spendable: 3 }[card && card.pace_status] || 4;
    var used = usedOf(card);
    return [0, group, Number.isFinite(used) ? used : Infinity, Infinity, index];
  }

  function sortCards(cards) {
    var list = (cards || []).slice();
    markUseNext(list);
    return list
      .map(function (card, index) {
        return { card: card, index: index };
      })
      .sort(function (a, b) {
        return compareKeys(displaySortKey(a.card, a.index), displaySortKey(b.card, b.index));
      })
      .map(function (row) {
        return row.card;
      });
  }

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // The first row of a subscription card's limits, drawn like the vendor
  // limit rows below it: label and value on one line, the bar under them.
  function rowHtml(card) {
    if (!isSubscriptionCard(card) || !card.pace_status) return "";
    var used = usedOf(card);
    var known = Number.isFinite(used);
    var fill = known ? Math.max(0, Math.min(SCALE, used)) : 0;
    // Colour follows the number shown: red over 100%, yellow over 85%, green below.
    var shown = known ? Math.round(used) : NaN;
    var zone = !known ? "unknown" : shown > SCALE ? "over" : shown > WARN ? "warn" : "ok";
    // A used-up limit is shown as the backend words it, not as a projection.
    var hit = !!(card.pace_limit_hit && card.pace_label);
    var value = hit ? card.pace_label : known ? formatPercent(used) : "—";
    var title = (card.pace_note ? card.pace_note + "\n\n" : "") + LEGEND;
    var classes = "usage-sub-pace-row is-" + zone + (card.pace_limit_hit ? " is-hit" : "");
    return (
      '<div class="' + classes + '" title="' + esc(title) + '">' +
        '<div class="usage-sub-row">' +
          '<span class="usage-sub-pace-label">Pace' +
            (card.pace_use_next ? ' <span class="usage-sub-use-next">Use next</span>' : "") +
          "</span>" +
          '<span class="usage-sub-pct usage-sub-pace-value">' +
            (hit ? '<span class="usage-sub-hit">' + esc(value) + "</span>" : esc(value)) +
          "</span>" +
        "</div>" +
        '<div class="usage-sub-bar" role="meter" aria-label="Pace ' + esc(value) + '"' +
          ' aria-valuemin="0" aria-valuemax="' + SCALE + '"' +
          ' aria-valuenow="' + esc(known ? String(used) : "0") + '">' +
          '<span style="width:' + fill + '%"></span>' +
        "</div>" +
      "</div>"
    );
  }

  function legendText() {
    return LEGEND;
  }

  root.saUsagePace = {
    scaleMax: scaleMax,
    formatPercent: formatPercent,
    rowHtml: rowHtml,
    legendText: legendText,
    markUseNext: markUseNext,
    sortCards: sortCards,
    isSubscriptionCard: isSubscriptionCard,
  };
})(typeof globalThis !== "undefined" ? globalThis : this);

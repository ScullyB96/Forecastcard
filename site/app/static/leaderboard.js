// Games/Players view toggle, per-game filter, and column sort for the
// slate-wide player leaderboard. One delegated listener covers every
// sport page -- no per-element listeners to wire up on render.
document.addEventListener("click", (event) => {
  const viewBtn = event.target.closest("[data-view]");
  if (viewBtn) {
    const toggle = viewBtn.closest("[data-view-toggle]");
    const scope = toggle.parentElement;
    toggle.querySelectorAll("[data-view]").forEach((b) => b.classList.toggle("active", b === viewBtn));
    scope.querySelectorAll("[data-view-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.viewPanel !== viewBtn.dataset.view;
    });
    return;
  }

  const chip = event.target.closest("[data-game-filter]");
  if (chip) {
    const bar = chip.closest(".game-filter-bar");
    const board = chip.closest("[data-leaderboard]");
    const gameId = chip.dataset.gameFilter;
    bar.querySelectorAll("[data-game-filter]").forEach((c) => c.classList.toggle("active", c === chip));
    board.querySelectorAll("tbody tr[data-game]").forEach((row) => {
      row.style.display = gameId === "all" || row.dataset.game === gameId ? "" : "none";
    });
    return;
  }

  const th = event.target.closest("[data-sort-key]");
  if (th) {
    const table = th.closest("table");
    const idx = parseInt(th.dataset.sortKey, 10);
    const numeric = th.dataset.sortType === "num";
    const desc = th.dataset.sortDir !== "desc";
    table.querySelectorAll("[data-sort-key]").forEach((h) => {
      delete h.dataset.sortDir;
      h.classList.remove("sort-asc", "sort-desc");
    });
    th.dataset.sortDir = desc ? "desc" : "asc";
    th.classList.add(desc ? "sort-desc" : "sort-asc");

    const tbody = table.querySelector("tbody");
    const rows = Array.from(tbody.querySelectorAll("tr"));
    rows.sort((a, b) => {
      const av = a.children[idx].dataset.value;
      const bv = b.children[idx].dataset.value;
      if (numeric) {
        const an = parseFloat(av);
        const bn = parseFloat(bv);
        const aNum = Number.isNaN(an) ? -Infinity : an;
        const bNum = Number.isNaN(bn) ? -Infinity : bn;
        return desc ? bNum - aNum : aNum - bNum;
      }
      return desc ? bv.localeCompare(av) : av.localeCompare(bv);
    });
    rows.forEach((r) => tbody.appendChild(r));
  }
});

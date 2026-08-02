// Filters a game card's props panel to one market at a time when a
// filter chip is clicked -- the same "pick a category" pattern real
// sportsbook prop pages (Action Network, DK) use instead of one long
// scroll. One delegated listener covers every card on the page.
document.addEventListener("click", (event) => {
  const chip = event.target.closest(".prop-filter-chip");
  if (!chip) return;

  const panel = chip.closest(".prop-panel");
  if (!panel) return;

  const market = chip.dataset.market;
  panel.querySelectorAll(".prop-filter-chip").forEach((c) => {
    c.classList.toggle("active", c === chip);
  });
  panel.querySelectorAll(".prop-market-group").forEach((group) => {
    group.style.display = market === "all" || group.dataset.market === market ? "" : "none";
  });
  panel.querySelectorAll("[data-role-section]").forEach((section) => {
    const anyVisible = Array.from(section.querySelectorAll(".prop-market-group")).some(
      (g) => g.style.display !== "none"
    );
    section.style.display = anyVisible ? "" : "none";
  });
});

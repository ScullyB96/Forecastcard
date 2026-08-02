function initCountdowns() {
  document.querySelectorAll("[data-countdown]").forEach(function (el) {
    var target = new Date(el.dataset.countdown).getTime();
    function tick() {
      var diff = target - Date.now();
      if (diff <= 0) {
        el.textContent = "Live!";
        return;
      }
      var d = Math.floor(diff / 86400000);
      var h = Math.floor((diff % 86400000) / 3600000);
      var m = Math.floor((diff % 3600000) / 60000);
      var s = Math.floor((diff % 60000) / 1000);
      el.textContent = d + "d " + h + "h " + m + "m " + s + "s";
    }
    tick();
    setInterval(tick, 1000);
  });
}
document.addEventListener("DOMContentLoaded", initCountdowns);

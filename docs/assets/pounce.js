// POUNCE mdBook theme menu polish.
// Reskins the built-in theme slots into a clean two-theme toggle:
//   navy  -> "Tiger" (dark, default)   light -> "Cream" (light)
// Hides the unused built-ins (rust/coal/ayu) and marks the active row.
(function () {
  function decorate() {
    var list = document.querySelector(".theme-popup, #theme-list");
    var navy = document.getElementById("mdbook-theme-navy");
    var light = document.getElementById("mdbook-theme-light");
    if (!navy || !light) return;

    navy.textContent = "Tiger (dark)";
    light.textContent = "Cream (light)";

    ["mdbook-theme-rust", "mdbook-theme-coal", "mdbook-theme-ayu"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) {
        var li = el.closest("li") || el;
        li.style.display = "none";
      }
    });

    // Put Tiger (the default) directly under "Auto", ahead of Cream.
    var navyLi = navy.closest("li") || navy;
    var lightLi = light.closest("li") || light;
    if (navyLi.parentNode && lightLi.parentNode === navyLi.parentNode) {
      navyLi.parentNode.insertBefore(navyLi, lightLi);
    }

    markActive();
  }

  function markActive() {
    var cls = document.documentElement.classList;
    document.querySelectorAll(".theme").forEach(function (b) {
      b.classList.remove("pounce-active");
    });
    var active = cls.contains("navy")
      ? "mdbook-theme-navy"
      : cls.contains("light")
      ? "mdbook-theme-light"
      : null;
    if (active) {
      var el = document.getElementById(active);
      if (el) el.classList.add("pounce-active");
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", decorate);
  } else {
    decorate();
  }
  // Re-mark when the user picks a theme (mdBook swaps the <html> class).
  document.addEventListener("click", function (e) {
    if (e.target && e.target.classList && e.target.classList.contains("theme")) {
      setTimeout(markActive, 0);
    }
  });
})();

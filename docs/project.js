document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("[data-copy]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var el = document.getElementById(btn.getAttribute("data-copy"));
      if (!el || !navigator.clipboard) return;
      navigator.clipboard.writeText(el.innerText).then(function () {
        var old = btn.textContent; btn.textContent = "Copied";
        var s = document.getElementById("copy-status"); if (s) s.textContent = "BibTeX copied";
        setTimeout(function () { btn.textContent = old; }, 1600);
      });
    });
  });
  if (window.katex) {
    document.querySelectorAll("span.math").forEach(function (el) {
      try { katex.render(el.textContent, el, { displayMode: el.classList.contains("display"), throwOnError: false }); } catch (e) {}
    });
  }
  if (window.renderMathInElement) {
    renderMathInElement(document.getElementById("main"), {
      delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }, { left: "$", right: "$", display: false }],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
    });
  }
  if (document.querySelector("pre.mermaid")) {
    var s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js";
    s.onload = function () { mermaid.initialize({ startOnLoad: false, theme: "neutral" }); mermaid.run({ querySelector: "pre.mermaid" }); };
    document.head.appendChild(s);
  }
});

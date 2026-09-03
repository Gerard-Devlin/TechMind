(() => {
  if (!("serviceWorker" in navigator)) return;

  const script = document.currentScript;
  if (!script?.src) return;

  const siteRoot = new URL("../", script.src);
  const workerUrl = new URL("service-worker.js", siteRoot);

  window.addEventListener("load", () => {
    navigator.serviceWorker.register(workerUrl, { scope: siteRoot.pathname }).catch(error => {
      console.warn("TechMind service worker registration failed:", error);
    });
  }, { once: true });
})();

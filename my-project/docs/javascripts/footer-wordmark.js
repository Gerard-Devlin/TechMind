/* Asciify's MIT-licensed wordmark renderer, with local assets and MkDocs cleanup. */
(() => {
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  let renderer;
  let cursorRenderer;
  let teardown = () => {};

  function mount() {
    teardown();
    teardown = () => {};
    const mark = document.querySelector("[data-tm-ascii-wordmark]");
    if (!mark || reducedMotion.matches || !("IntersectionObserver" in window)) return;

    const art = mark.querySelector(".tm-ascii-wordmark__art");
    let disposed = false;
    let handle;
    let cursor;
    let fadeTimer;
    let leaveTimer;
    let width = mark.clientWidth;
    const ink = () => getComputedStyle(mark).getPropertyValue("--tm-ascii-ink").trim();
    const look = () => ({
      font: '"TechMind ASCII Glyph", Arial, sans-serif',
      cellSize: Math.max(4, Math.min(8, mark.clientWidth / 190)),
      hover: "none"
    });
    const showStill = () => mark.removeAttribute("data-live");
    const cursorOptions = () => {
      const box = art.getBoundingClientRect();
      return {
        ink: ink(),
        cellSize: look().cellSize,
        look: { font: '"TechMind ASCII Glyph", Arial, sans-serif' },
        trail: { span: Math.min(box.width, box.height), cell: Math.max(box.width, box.height) / 128 }
      };
    };
    const clipCursor = () => {
      const overlay = cursor?.field?.canvas?.parentElement ?? document.querySelector("[data-ascii-cursor]");
      if (!overlay) return;
      const box = mark.getBoundingClientRect();
      overlay.style.clipPath = `inset(${Math.max(0, box.top)}px ${Math.max(0, innerWidth - box.right)}px ${Math.max(0, innerHeight - box.bottom)}px ${Math.max(0, box.left)}px)`;
    };
    const enter = () => { clearTimeout(leaveTimer); cursor?.play(); };
    const leave = () => { leaveTimer = setTimeout(() => cursor?.pause(), 1000); };
    const resize = new ResizeObserver(() => {
      if (width === mark.clientWidth) return;
      width = mark.clientWidth;
      handle?.update({ look: look() });
      cursor?.update(cursorOptions());
      clipCursor();
    });
    const palette = new MutationObserver(() => {
      handle?.update({ ink: ink() });
      cursor?.update({ ink: ink() });
    });
    palette.observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
    resize.observe(mark);
    mark.addEventListener("pointerenter", enter);
    mark.addEventListener("pointerleave", leave);
    mark.addEventListener("pointermove", clipCursor);
    window.addEventListener("scroll", clipCursor, { passive: true });

    async function activate() {
      try {
        renderer ??= import(new URL("../assets/footer-ascii/wordmark.js", import.meta.url));
        const [module] = await Promise.all([
          renderer,
          document.fonts.load('600 100px "TechMind ASCII Serif"', "TechMind"),
          document.fonts.load('500 16px "TechMind ASCII Glyph"', " .:-=+*#%@")
        ]);
        if (disposed || !mark.isConnected) return;
        handle = module.N(art, {
          text: "TechMind",
          font: '"TechMind ASCII Serif", Georgia, serif',
          fontWeight: 600,
          label: "TechMind",
          ink: ink(),
          look: look(),
          onFail: showStill
        });
        await handle.field?.ready;
        if (disposed || !handle.supported) return;
        fadeTimer = setTimeout(() => {
          if (!disposed && handle.supported) mark.setAttribute("data-live", "");
        }, 180);
        if (matchMedia("(hover: hover) and (pointer: fine)").matches) {
          cursorRenderer ??= import(new URL("../assets/footer-ascii/cursor-trail.js", import.meta.url));
          const { cursorTrail } = await cursorRenderer;
          if (disposed) return;
          cursor = cursorTrail({ style: "trail", readout: true, zIndex: 1, ...cursorOptions() });
          if (!mark.matches(":hover")) cursor.pause();
        }
      } catch (error) {
        showStill();
        // The static local SVG remains usable if WebGL or the lazy module fails.
        console.warn("TechMind footer uses its static ASCII fallback:", error);
      }
    }

    const nearby = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting)) return;
      nearby.disconnect();
      activate();
    }, { rootMargin: "50% 0px" });
    nearby.observe(mark);

    teardown = () => {
      disposed = true;
      clearTimeout(fadeTimer);
      clearTimeout(leaveTimer);
      nearby.disconnect();
      resize.disconnect();
      palette.disconnect();
      mark.removeEventListener("pointerenter", enter);
      mark.removeEventListener("pointerleave", leave);
      mark.removeEventListener("pointermove", clipCursor);
      window.removeEventListener("scroll", clipCursor);
      cursor?.destroy();
      handle?.destroy();
      showStill();
    };
  }

  reducedMotion.addEventListener("change", mount);
  if (typeof document$ !== "undefined") document$.subscribe(mount);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount, { once: true });
  else mount();
})();

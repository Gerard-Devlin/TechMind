/* Follow Material's existing scroll tracking; only add the moving TOC marker. */
(() => {
  let dispose = () => {};

  function mount() {
    dispose();
    dispose = () => {};
    const rail = document.querySelector('.md-sidebar--secondary [data-md-component="toc"]');
    if (!rail) return;
    const links = [...rail.querySelectorAll('.md-nav__link')];
    if (!links.length) return;

    let active = null;
    let frame = 0;

    function update() {
      frame = 0;
      const current = links.find(link => link.classList.contains('md-nav__link--active')) || links[0];
      if (current !== active) {
        active?.removeAttribute('aria-current');
        current.setAttribute('aria-current', 'location');
        active = current;
      }
      // The root list is the positioned ancestor, including for nested entries.
      rail.style.setProperty('--tm-toc-offset', `${current.offsetTop}px`);
      rail.style.setProperty('--tm-toc-height', `${current.offsetHeight}px`);
      rail.dataset.tmTocReady = 'true';
    }

    function schedule() {
      if (!frame) frame = requestAnimationFrame(update);
    }

    const mutationObserver = new MutationObserver(schedule);
    mutationObserver.observe(rail, { subtree: true, attributes: true, attributeFilter: ['class'] });
    const resizeObserver = new ResizeObserver(schedule);
    resizeObserver.observe(rail);
    links.forEach(link => resizeObserver.observe(link));
    window.addEventListener('resize', schedule, { passive: true });
    schedule();

    dispose = () => {
      mutationObserver.disconnect();
      resizeObserver.disconnect();
      window.removeEventListener('resize', schedule);
      cancelAnimationFrame(frame);
      active?.removeAttribute('aria-current');
      delete rail.dataset.tmTocReady;
    };
  }

  if (typeof document$ !== 'undefined') document$.subscribe(mount);
  else if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, { once: true });
  else mount();
})();

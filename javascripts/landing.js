/* Home-only choreography. Material's instant navigation reuses this script. */
(() => {
  let dispose = () => {};

  function mount() {
    dispose();
    const home = document.querySelector('.tm-landing');
    if (!home) return;

    const controller = new AbortController();
    const { signal } = controller;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    const desktop = matchMedia('(min-width: 1024px)');
    const cleanups = [];
    const behavior = () => reduced.matches ? 'instant' : 'smooth';
    const clamp = (n, min = 0, max = 1) => Math.min(max, Math.max(min, n));

    home.querySelectorAll('[data-tm-carousel]').forEach(carousel => {
      const track = carousel.querySelector('.tm-carousel__track');
      const prev = carousel.querySelector('[data-tm-prev]');
      const next = carousel.querySelector('[data-tm-next]');
      const update = () => {
        prev.disabled = track.scrollLeft <= 2;
        next.disabled = track.scrollLeft >= track.scrollWidth - track.clientWidth - 2;
      };
      const move = direction => {
        const gap = parseFloat(getComputedStyle(track).columnGap) || 0;
        const distance = track.firstElementChild.getBoundingClientRect().width + gap;
        track.scrollBy({ left: direction * distance, behavior: behavior() });
      };
      prev.addEventListener('click', () => move(-1), { signal });
      next.addEventListener('click', () => move(1), { signal });
      track.addEventListener('scroll', update, { passive: true, signal });
      track.addEventListener('keydown', event => {
        if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
        event.preventDefault();
        move(event.key === 'ArrowLeft' ? -1 : 1);
      }, { signal });
      const observer = new ResizeObserver(update);
      observer.observe(track);
      cleanups.push(() => observer.disconnect());
      update();
    });

    const showcase = home.querySelector('[data-tm-showcase]');
    if (showcase) {
      const sticky = showcase.querySelector('.tm-showcase__sticky');
      const artwork = showcase.querySelector('.tm-showcase__art');
      const steps = [...showcase.querySelectorAll('[data-tm-step]')];
      const frames = [...showcase.querySelectorAll('[data-tm-frame]')];
      const stepList = showcase.querySelector('.tm-showcase__steps');
      const hero = home.querySelector('.tm-hero');
      let active = -1;
      let raf = 0;
      let start = 0;
      let distance = 1;
      let stickyTop = 80;
      let heroStart = 0;
      let heroHeight = 1;
      showcase.dataset.enhanced = 'true';

      function activate(index) {
        if (active === index) return;
        active = index;
        showcase.dataset.active = String(index);
        steps.forEach((step, i) => {
          step.classList.toggle('is-active', i === index);
          step.querySelector('button').setAttribute('aria-expanded', String(i === index));
          step.querySelector('.tm-showcase__detail').inert = i !== index;
          frames[i].classList.toggle('is-active', i === index);
          frames[i].classList.toggle('is-past', i < index);
        });
      }

      function render() {
        raf = 0;
        const drift = reduced.matches ? 0 : clamp((scrollY - heroStart) / heroHeight) * 24;
        hero.style.setProperty('--tm-hero-drift', `${drift.toFixed(2)}px`);
        if (!desktop.matches) return;
        if (reduced.matches) {
          showcase.style.setProperty('--tm-shift', '1');
          showcase.style.setProperty('--tm-step-opacity', '1');
          stepList.inert = false;
          if (active < 0) activate(0);
          return;
        }
        const progress = clamp((scrollY - start) / distance);
        const entrance = clamp(progress / .2);
        const shift = entrance * entrance * (3 - 2 * entrance);
        const sequence = clamp((progress - .2) / .8) * 3;
        const index = Math.min(2, Math.floor(sequence));
        showcase.style.setProperty('--tm-shift', shift.toFixed(4));
        showcase.style.setProperty('--tm-step-opacity', clamp((shift - .4) / .6).toFixed(4));
        // Invisible copy must not intercept clicks or keyboard focus during the introduction.
        stepList.inert = shift < .75;
        activate(index);
        steps.forEach((step, i) => step.style.setProperty('--tm-step-progress', String(clamp(sequence - i))));
      }

      function schedule() {
        if (!raf) raf = requestAnimationFrame(render);
      }

      function measure() {
        showcase.dataset.scrollMotion = String(desktop.matches && !reduced.matches);
        heroStart = hero.getBoundingClientRect().top + scrollY;
        heroHeight = Math.max(1, hero.offsetHeight);
        stickyTop = parseFloat(getComputedStyle(showcase).getPropertyValue('--tm-sticky-top')) || 80;
        start = showcase.getBoundingClientRect().top + scrollY - stickyTop;
        distance = Math.max(1, showcase.offsetHeight - sticky.offsetHeight);
        stepList.inert = false;
        if (desktop.matches) schedule();
        else {
          const width = frames[0].getBoundingClientRect().width + 24;
          activate(clamp(Math.round(artwork.scrollLeft / width), 0, 2));
          schedule();
        }
      }

      steps.forEach((step, index) => {
        step.querySelector('button').addEventListener('click', () => {
          if (desktop.matches && !reduced.matches) {
            window.scrollTo({ top: start + distance * (.2 + (index + .15) * .8 / 3), behavior: behavior() });
          } else if (desktop.matches) {
            activate(index);
          } else {
            const width = frames[0].getBoundingClientRect().width + 24;
            artwork.scrollTo({ left: index * width, behavior: behavior() });
            activate(index);
          }
        }, { signal });
      });
      artwork.addEventListener('scroll', () => {
        if (desktop.matches) return;
        const width = frames[0].getBoundingClientRect().width + 24;
        activate(clamp(Math.round(artwork.scrollLeft / width), 0, 2));
      }, { signal, passive: true });
      window.addEventListener('scroll', schedule, { signal, passive: true });
      window.addEventListener('resize', measure, { signal, passive: true });
      desktop.addEventListener('change', measure, { signal });
      reduced.addEventListener('change', measure, { signal });
      const observer = new ResizeObserver(measure);
      observer.observe(hero);
      cleanups.push(() => { observer.disconnect(); cancelAnimationFrame(raf); });
      measure();
    }

    dispose = () => {
      controller.abort();
      cleanups.forEach(cleanup => cleanup());
    };
  }

  if (typeof document$ !== 'undefined') document$.subscribe(mount);
  else if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, { once: true });
  else mount();
})();

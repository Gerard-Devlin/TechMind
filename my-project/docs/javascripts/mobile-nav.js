/* OpenAI.com-style panel navigation, with MkDocs' generated course tree. */
(() => {
  const dialog = document.querySelector('.tm-mobile-nav');
  const toggle = document.querySelector('#__drawer');
  const trigger = document.querySelector('.md-header label[for="__drawer"]');
  if (!dialog || !toggle || !trigger) return;

  const mobile = matchMedia('(max-width: 76.234375em)');
  const scroll = dialog.querySelector('.tm-mobile-nav__scroll');
  const panels = [...dialog.querySelectorAll('[data-tm-nav-panel]')];
  const positions = new Map();
  let current = 'root';
  let returnFocus = trigger;
  let restoreFocus = true;
  trigger.tabIndex = 0;
  trigger.setAttribute('role', 'button');
  trigger.setAttribute('aria-label', 'Open navigation');
  trigger.setAttribute('aria-haspopup', 'dialog');
  trigger.setAttribute('aria-controls', dialog.id);
  trigger.setAttribute('aria-expanded', 'false');

  function showPanel(id, focus = true) {
    const next = panels.find(panel => panel.dataset.tmNavPanel === id);
    if (!next) return;
    positions.set(current, scroll.scrollTop);
    const previous = current;
    panels.forEach(panel => { panel.hidden = panel !== next; });
    dialog.querySelectorAll('[aria-controls][data-tm-nav-target]').forEach(button => {
      button.setAttribute('aria-expanded', String(button.dataset.tmNavTarget === id));
    });
    current = id;
    scroll.scrollTop = positions.get(id) || 0;
    if (focus) {
      const returnButton = [...next.querySelectorAll('[data-tm-nav-target]')].find(button => button.dataset.tmNavTarget === previous);
      (returnButton || next.querySelector('button, a'))?.focus({ preventScroll: true });
    }
  }

  function close(restore = true) {
    restoreFocus = restore;
    if (dialog.open) dialog.close();
    document.documentElement.classList.remove('tm-mobile-nav-open');
    if (toggle.checked) {
      toggle.checked = false;
      toggle.dispatchEvent(new Event('change', { bubbles: true }));
    }
  }

  function sync() {
    trigger.setAttribute('aria-expanded', String(toggle.checked && mobile.matches));
    if (toggle.checked && mobile.matches) {
      if (!dialog.open) {
        returnFocus = document.activeElement;
        positions.clear();
        showPanel('root', false);
        scroll.scrollTop = 0;
        document.documentElement.classList.add('tm-mobile-nav-open');
        dialog.showModal();
      }
    } else if (dialog.open) close();
  }

  toggle.addEventListener('change', sync);
  trigger.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      trigger.click();
    }
  });
  dialog.addEventListener('close', () => {
    document.documentElement.classList.remove('tm-mobile-nav-open');
    if (toggle.checked) {
      toggle.checked = false;
      toggle.dispatchEvent(new Event('change', { bubbles: true }));
    }
    if (restoreFocus) {
      const target = returnFocus?.isConnected && returnFocus.getClientRects().length ? returnFocus : trigger;
      target.focus({ preventScroll: true });
    }
    restoreFocus = true;
  });
  dialog.addEventListener('click', event => {
    if (event.target.closest('[data-tm-nav-close]')) close();
    else if (event.target.closest('[data-tm-nav-target]')) showPanel(event.target.closest('[data-tm-nav-target]').dataset.tmNavTarget);
    else if (event.target.closest('[data-tm-nav-search]')) {
      close(false);
      requestAnimationFrame(() => document.querySelector('.tm-search__trigger')?.click());
    } else if (event.target.closest('a[href]')) close(false);
  });
  mobile.addEventListener('change', () => { if (!mobile.matches) close(false); });

  function update() {
    close(false);
    dialog.querySelectorAll('a[href]').forEach(link => {
      const url = new URL(link.href);
      const active = url.origin === location.origin && url.pathname === location.pathname && !url.hash;
      if (active) {
        link.setAttribute('aria-current', 'page');
      } else link.removeAttribute('aria-current');
    });
  }
  if (typeof document$ !== 'undefined') document$.subscribe(update);
  else update();
})();

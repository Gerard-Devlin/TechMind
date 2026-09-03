/* Extend the native search controls; Material still owns queries and results. */
(() => {
  const search = document.querySelector('.tm-search');
  const toggle = document.querySelector('[data-md-toggle="search"]');
  if (!search || !toggle) return;

  const panel = search.querySelector('.md-search__inner');
  const query = search.querySelector('[data-md-component="search-query"]');
  const trigger = search.querySelector('.tm-search__trigger');
  const mobileTrigger = document.querySelector('.md-header__button[for="__search"]');
  let returnFocus = trigger;

  if (/Mac|iPhone|iPad/.test(navigator.platform)) {
    search.querySelector('[data-tm-search-shortcut]').textContent = '⌘ K';
  }

  function open() {
    returnFocus = document.activeElement;
    if (!toggle.checked) toggle.click();
    query.focus();
    query.select();
  }

  function close() {
    if (toggle.checked) toggle.click();
    query.blur();
    const target = returnFocus?.isConnected && returnFocus !== query ? returnFocus : trigger;
    if (target.getClientRects().length) target.focus();
    else mobileTrigger?.focus();
  }

  function sync() {
    if (!toggle.checked && panel.contains(document.activeElement)) document.activeElement.blur();
    panel.setAttribute('aria-hidden', String(!toggle.checked));
    trigger.setAttribute('aria-expanded', String(toggle.checked));
    mobileTrigger?.setAttribute('aria-expanded', String(toggle.checked));
    query.tabIndex = toggle.checked ? 0 : -1;
  }

  trigger.addEventListener('click', open);
  if (mobileTrigger) {
    mobileTrigger.tabIndex = 0;
    mobileTrigger.setAttribute('role', 'button');
    mobileTrigger.setAttribute('aria-haspopup', 'dialog');
    mobileTrigger.setAttribute('aria-controls', panel.id);
    mobileTrigger.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        event.stopPropagation();
        open();
      }
    });
  }
  search.querySelector('.md-search__overlay').addEventListener('click', event => {
    event.preventDefault();
    close();
  });
  toggle.addEventListener('change', sync);
  sync();

  // Capture Tab before Material's default "Tab closes search" handler so the
  // palette has a real focus loop, including clear / share / close controls.
  document.addEventListener('keydown', event => {
    if (event.isComposing) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      toggle.checked ? close() : open();
      return;
    }
    // The native /, s and f shortcuts focus the input to open search. Reveal
    // it first so its hidden dialog is never the accessibility focus target.
    if (!toggle.checked && !event.ctrlKey && !event.metaKey && !event.altKey &&
        ['/', 's', 'f'].includes(event.key) &&
        !event.target.closest('input, textarea, select, [contenteditable="true"]')) {
      event.preventDefault();
      event.stopPropagation();
      open();
      return;
    }
    if (!toggle.checked) return;
    if (event.key === ' ' && event.target.closest('button, summary')) {
      event.stopPropagation();
      return;
    }
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      close();
    } else if (event.key === 'Tab') {
      event.preventDefault();
      event.stopPropagation();
      const focusable = [...panel.querySelectorAll('input, button, a[href], summary')]
        .filter(element => element.getClientRects().length &&
          getComputedStyle(element).visibility !== 'hidden' &&
          getComputedStyle(element).opacity !== '0');
      const index = focusable.indexOf(document.activeElement);
      const next = (index + (event.shiftKey ? -1 : 1) + focusable.length) % focusable.length;
      focusable[next]?.focus();
    }
  }, true);
})();

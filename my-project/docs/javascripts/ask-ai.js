import { Marked, Renderer } from '../assets/vendor/ask-ai/marked.esm.js';
import DOMPurify from '../assets/vendor/ask-ai/purify.es.mjs';
import { courseImagePath, readEvents, requestHistory, sourcePageTitle, trustedDocURL, validatedNavigation, validatedSources } from './ask-ai-core.js?v=20261023-6';

const root = document.querySelector('[data-tm-ai-root]');
if (root && !root.dataset.initialized) initialize(root);

function initialize(root) {
  root.dataset.initialized = 'true';
  const $ = selector => root.querySelector(selector);
  const panel = $('.tm-ai__panel');
  const launcher = $('[data-ai-open]');
  const thread = $('[data-ai-thread]');
  const messages = $('[data-ai-messages]');
  const welcome = $('[data-ai-welcome]');
  const composer = $('[data-ai-form]');
  const input = $('[data-ai-input]');
  const send = $('[data-ai-send]');
  const notice = $('[data-ai-notice]');
  const modeLabel = $('[data-ai-mode]');
  const mobile = matchMedia('(max-width: 767px)');
  const siteBase = new URL('../', import.meta.url);
  const docsBase = root.dataset.docsBase;
  let courseImages = {};
  try {
    const inventory = JSON.parse(root.dataset.courseImages || '{}');
    if (inventory && typeof inventory === 'object' && !Array.isArray(inventory)) courseImages = inventory;
  } catch {}
  const storageKey = 'techmind.docs-agent.v1';
  const navigationStorageKey = 'techmind.docs-agent.navigation.v1';
  const clientStorageKey = 'techmind.docs-agent.client.v1';
  let clientSession;
  try {
    const saved = JSON.parse(sessionStorage.getItem(clientStorageKey) || 'null');
    if (saved?.expires > Date.now() && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(saved.id)) clientSession = saved;
  } catch {}
  function clientID() {
    if (!clientSession || clientSession.expires <= Date.now()) {
      clientSession = { id: crypto.randomUUID(), expires: Date.now() + 30 * 60 * 1000 };
      try { sessionStorage.setItem(clientStorageKey, JSON.stringify(clientSession)); } catch {}
    }
    return clientSession.id;
  }
  const markdownRenderer = new Renderer();
  markdownRenderer.html = () => '';
  const markdown = new Marked({ gfm: true, breaks: false, async: false, renderer: markdownRenderer });
  const escapeHTML = text => text.replace(/[&<>"']/g, value => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[value]);
  // Protect TeX before Markdown treats \( and \[ as escaped punctuation. Code
  // fences/code spans keep their own lexer precedence and are never typeset.
  markdown.use({ extensions: [{
    name: 'techmindMath', level: 'inline',
    start(source) { const match = /\\[([]|\$/.exec(source); return match?.index; },
    tokenizer(source) {
      const match = /^(?:\\\(([\s\S]+?)\\\)|\\\[([\s\S]+?)\\\]|\$\$([\s\S]+?)\$\$|\$(?!\$)([^\n$]+?)\$(?!\$))/.exec(source);
      if (!match) return undefined;
      return { type: 'techmindMath', raw: match[0], text: match[1] ?? match[2] ?? match[3] ?? match[4], display: match[2] !== undefined || match[3] !== undefined };
    },
    renderer(token) { return `<code class="tm-ai__math-${token.display ? 'display' : 'inline'}">${escapeHTML(token.text)}</code>`; },
  }] });
  const arrowIcon = '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 19V5m-5 5 5-5 5 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const copyIcon = '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="8" y="3" width="12" height="14" rx="3" stroke="currentColor" stroke-width="1.6"/><path d="M16 17v1a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3v-8a3 3 0 0 1 3-3h1" stroke="currentColor" stroke-width="1.6"/></svg>';
  let turns = [];
  let selectedMode = 'answer';
  let controller = null;
  let operation = 0;
  let returnFocus = launcher;
  let pinnedToBottom = true;
  let renderFrame = null;
  let scrollFrame = null;
  let restoredScrollPending = false;
  let turnstilePromise = null;

  function announce(text) { $('[data-ai-announcement]').textContent = text; }
  function setNotice(text = '') { notice.textContent = text; notice.hidden = !text; }
  function setMode(mode) {
    selectedMode = ['answer', 'find', 'guide'].includes(mode) ? mode : 'answer';
    modeLabel.hidden = selectedMode === 'answer';
    modeLabel.querySelector('span').textContent = selectedMode === 'find' ? 'Find a page' : 'Build a custom guide';
    input.placeholder = selectedMode === 'find' ? 'Which topic are you looking for?' : selectedMode === 'guide' ? 'What would you like to learn?' : 'Ask about docs or what you want to build';
    input.setAttribute('aria-label', input.placeholder);
  }

  function open() {
    returnFocus = document.activeElement;
    root.classList.add('is-open');
    document.body.classList.add('tm-ai-open');
    launcher.setAttribute('aria-expanded', 'true');
    panel.setAttribute('aria-hidden', 'false');
    panel.inert = false;
    // Do not summon a phone keyboard just by opening the drawer.
    (mobile.matches ? $('[data-ai-close]') : input).focus({ preventScroll: true });
    positionRestoredConversation();
    scheduleFollowScroll();
    if (root.dataset.siteKey) loadTurnstile().catch(() => {});
  }

  function close() {
    root.classList.remove('is-open');
    document.body.classList.remove('tm-ai-open');
    launcher.setAttribute('aria-expanded', 'false');
    panel.setAttribute('aria-hidden', 'true');
    panel.inert = true;
    const target = returnFocus?.isConnected && !panel.contains(returnFocus) ? returnFocus : launcher;
    target.focus({ preventScroll: true });
  }

  function updateInput() {
    input.style.height = '36px';
    input.style.height = `${Math.min(126, Math.max(36, input.scrollHeight))}px`;
    send.disabled = !controller && !input.value.trim();
  }

  function updateBusy(busy) {
    send.innerHTML = busy ? '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6.5" y="6.5" width="11" height="11" rx="1.5"/></svg>' : arrowIcon;
    send.setAttribute('aria-label', busy ? 'Stop response' : 'Send message');
    send.type = busy ? 'button' : 'submit';
    send.disabled = !busy && !input.value.trim();
    panel.setAttribute('aria-busy', String(busy));
  }

  function atBottom() { return thread.scrollHeight - thread.scrollTop - thread.clientHeight < 80; }
  function followScroll(force = false) {
    if (force || pinnedToBottom) thread.scrollTop = thread.scrollHeight;
    $('[data-ai-bottom]').hidden = atBottom();
  }
  function positionRestoredConversation() {
    if (!restoredScrollPending || !root.classList.contains('is-open')) return;
    restoredScrollPending = false;
    pinnedToBottom = true;
    followScroll(true);
  }
  function scheduleFollowScroll() {
    if (scrollFrame !== null) return;
    scrollFrame = requestAnimationFrame(() => {
      scrollFrame = null;
      // Read the current state at paint time: a user who has meanwhile scrolled
      // up or closed the panel must not be pulled back by an old layout task.
      if (root.classList.contains('is-open')) followScroll();
    });
  }
  function currentPage() {
    const relative = location.pathname.startsWith(siteBase.pathname) ? location.pathname.slice(siteBase.pathname.length) : '';
    const url = trustedDocURL(`${relative}${location.hash}`, docsBase);
    return url || undefined;
  }

  function localDocumentURL(canonical) {
    const safe = trustedDocURL(canonical, docsBase);
    if (!safe) return null;
    const url = new URL(safe);
    const base = new URL(docsBase);
    return new URL(`${url.pathname.slice(base.pathname.length)}${url.hash}`, siteBase).href;
  }

  function prepareDocumentNavigation(canonical) {
    const target = localDocumentURL(canonical);
    if (!target) return null;
    save();
    if (mobile.matches) close();
    else {
      try { sessionStorage.setItem(navigationStorageKey, JSON.stringify({ target, expires: Date.now() + 30000 })); } catch {}
    }
    return target;
  }

  function save() {
    // Session-only, bounded, no server-side chat store. New chat clears it.
    try {
      const saved = turns.filter(turn => turn.complete).slice(-6).map(({ question, answer, mode, sources, vote }) => ({ question, answer, mode, sources, vote }));
      const value = JSON.stringify({ expires: Date.now() + 30 * 60 * 1000, turns: saved });
      if (value.length <= 60000) sessionStorage.setItem(storageKey, value);
    } catch {}
  }

  function renderMarkdown(turn, container) {
    try { renderRichMarkdown(turn, container); }
    catch { container.textContent = turn.answer; }
  }

  function renderRichMarkdown(turn, container) {
    const html = markdown.parse(turn.answer);
    const fragment = DOMPurify.sanitize(html, {
      RETURN_DOM_FRAGMENT: true,
      ALLOWED_TAGS: ['p','br','strong','em','del','s','a','ul','ol','li','blockquote','pre','code','h1','h2','h3','h4','hr','table','thead','tbody','tr','th','td'],
      ALLOWED_ATTR: ['href','title','start','colspan','rowspan','class'],
      ALLOW_DATA_ATTR: false,
      ALLOW_ARIA_ATTR: false,
    });
    for (const anchor of fragment.querySelectorAll('a')) {
      const url = trustedDocURL(anchor.getAttribute('href'), docsBase);
      const source = turn.sources.find(item => item.url === url);
      if (!source) anchor.replaceWith(...anchor.childNodes);
      else { anchor.href = localDocumentURL(source.url); anchor.dataset.aiDocument = source.url; }
    }
    for (const math of fragment.querySelectorAll('code.tm-ai__math-display, code.tm-ai__math-inline')) {
      const span = document.createElement('span');
      span.className = 'tm-ai__math';
      span.textContent = math.classList.contains('tm-ai__math-display') ? `\\[${math.textContent}\\]` : `\\(${math.textContent}\\)`;
      math.replaceWith(span);
    }
    const walker = document.createTreeWalker(fragment, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (!node.parentElement?.closest('a,pre,code,.tm-ai__math') && /\[\d{1,2}(?:\s*,\s*\d{1,2})*\]/.test(node.textContent)) nodes.push(node);
    }
    for (const node of nodes) {
      const content = document.createDocumentFragment();
      const pattern = /\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g;
      let offset = 0;
      for (const match of node.textContent.matchAll(pattern)) {
        content.append(node.textContent.slice(offset, match.index));
        const cited = match[1].split(',').map(Number).map(id => turn.sources.find(source => source.id === id)).filter(Boolean);
        if (!cited.length) content.append(match[0]);
        for (const source of cited) {
          const anchor = document.createElement('a');
          anchor.className = 'tm-ai__citation';
          anchor.textContent = source.id;
          anchor.href = localDocumentURL(source.url);
          anchor.dataset.aiDocument = source.url;
          anchor.title = source.title;
          anchor.setAttribute('aria-label', `Source ${source.id}: ${source.title}`);
          content.append(anchor);
        }
        offset = match.index + match[0].length;
      }
      content.append(node.textContent.slice(offset));
      node.replaceWith(content);
    }
    for (const block of fragment.querySelectorAll('pre')) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'tm-ai__code-copy';
      button.textContent = 'Copy code';
      button.dataset.aiCopyCode = '';
      block.prepend(button);
    }
    for (const table of fragment.querySelectorAll('table')) {
      const wrapper = document.createElement('div');
      wrapper.className = 'tm-ai__table-wrap';
      table.replaceWith(wrapper); wrapper.append(table);
    }
    container.replaceChildren(fragment);
  }

  function sourceCard(source) {
    const card = document.createElement('div');
    card.className = 'tm-ai__source-card';
    const label = document.createElement('small');
    label.textContent = 'Related page';
    const title = document.createElement('h3');
    title.textContent = sourcePageTitle(source.title);
    title.title = source.title;
    title.setAttribute('aria-label', source.title);
    card.append(label);
    const imageFrame = document.createElement('div');
    imageFrame.className = 'tm-ai__source-image';
    const image = document.createElement('img');
    image.src = new URL(courseImagePath(source.url, docsBase, courseImages), siteBase).href;
    image.alt = ''; image.loading = 'lazy'; image.decoding = 'async';
    image.addEventListener('error', () => imageFrame.remove(), { once: true });
    imageFrame.append(image); card.append(imageFrame);
    const content = document.createElement('div'); content.className = 'tm-ai__source-content';
    content.append(title);
    if (source.excerpt) { const excerpt = document.createElement('p'); excerpt.textContent = source.excerpt; content.append(excerpt); }
    const origin = document.createElement('small'); origin.className = 'tm-ai__source-badge'; origin.textContent = 'TechMind docs';
    content.append(origin);
    const open = document.createElement('a'); open.className = 'tm-ai__open-page'; open.textContent = 'Open page';
    open.href = localDocumentURL(source.url); open.dataset.aiDocument = source.url;
    open.setAttribute('aria-label', `Open page: ${source.title}`);
    card.append(content, open);
    return card;
  }

  function renderSources(turn) {
    const area = turn.element.querySelector('[data-ai-sources]');
    area.replaceChildren();
    if (!turn.sources.length) return;
    // The original answer ends with one visible Related page, not an invented
    // Sources accordion or a stack of duplicate chapter cards. All retrieved
    // sections remain reachable through the answer's numbered citations; find
    // mode also lists its page candidates in the answer itself.
    area.append(sourceCard(turn.sources[0]));
  }

  function makeAction(label, icon, extra = {}) {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'tm-ai__icon-button';
    button.setAttribute('aria-label', label); button.title = label; button.innerHTML = icon;
    for (const [key, value] of Object.entries(extra)) button.dataset[key] = value;
    return button;
  }

  function renderActions(turn) {
    const actions = turn.element.querySelector('[data-ai-actions]');
    actions.replaceChildren();
    if (!turn.answer || !turn.complete) return;
    const copy = makeAction('Copy response', copyIcon, { aiCopy: '' });
    const like = makeAction('Helpful (saved on this device only)', '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M7 10v10H4a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1h3Zm0 0 5-7h1a2 2 0 0 1 2 2l-1 5h5a2 2 0 0 1 2 2l-2 7a2 2 0 0 1-2 1H7" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>', { aiVote: 'up' });
    const dislike = makeAction('Not helpful (saved on this device only)', '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><g transform="rotate(180 12 12)"><path d="M7 10v10H4a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1h3Zm0 0 5-7h1a2 2 0 0 1 2 2l-1 5h5a2 2 0 0 1 2 2l-2 7a2 2 0 0 1-2 1H7" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></g></svg>', { aiVote: 'down' });
    like.setAttribute('aria-pressed', String(turn.vote === 'up'));
    dislike.setAttribute('aria-pressed', String(turn.vote === 'down'));
    actions.append(copy, like, dislike);
  }

  function createTurn(question, mode, previous = {}) {
    const turn = { question, mode, answer: '', sources: [], complete: false, ...previous };
    const article = document.createElement('article');
    article.className = 'tm-ai__turn';
    const user = document.createElement('p'); user.className = 'tm-ai__user'; user.textContent = question;
    const status = document.createElement('p'); status.className = 'tm-ai__status'; status.dataset.aiStatus = ''; status.setAttribute('role', 'status');
    status.textContent = mode === 'find' ? 'Finding pages…' : 'Searching the docs…';
    const answer = document.createElement('div'); answer.className = 'tm-ai__answer'; answer.dataset.aiAnswer = '';
    const sources = document.createElement('div'); sources.className = 'tm-ai__sources'; sources.dataset.aiSources = '';
    const actions = document.createElement('div'); actions.className = 'tm-ai__actions'; actions.dataset.aiActions = '';
    article.append(user, status, answer, sources, actions);
    messages.append(article); turn.element = article;
    welcome.hidden = true;
    if (turn.complete) { status.hidden = true; renderMarkdown(turn, answer); renderSources(turn); renderActions(turn); renderMath(turn); }
    return turn;
  }

  function renderAnswer(turn, final = false) {
    const paint = () => {
      renderFrame = null;
      renderMarkdown(turn, turn.element.querySelector('[data-ai-answer]'));
      followScroll();
    };
    if (final) { if (renderFrame) cancelAnimationFrame(renderFrame); paint(); renderMath(turn); }
    else if (!renderFrame) renderFrame = requestAnimationFrame(paint);
  }

  function renderMath(turn) {
    const target = turn.element.querySelector('[data-ai-answer]');
    // Existing local page math runtime is reused only after sanitized Markdown.
    if (typeof window.renderMathInElement === 'function') {
      window.renderMathInElement(target, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }, { left: '$', right: '$', display: false }], throwOnError: false, trust: false, strict: 'ignore' });
    }
  }

  async function loadTurnstile() {
    if (window.turnstile) return window.turnstile;
    if (!turnstilePromise) turnstilePromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
      script.async = true;
      const timeout = setTimeout(() => reject(new Error('verification_unavailable')), 15000);
      script.onload = () => { clearTimeout(timeout); window.turnstile ? resolve(window.turnstile) : reject(new Error('verification_unavailable')); };
      script.onerror = () => { clearTimeout(timeout); reject(new Error('verification_unavailable')); };
      document.head.append(script);
    }).catch(error => { turnstilePromise = null; throw error; });
    return turnstilePromise;
  }

  async function verificationToken(signal) {
    if (!root.dataset.siteKey) return undefined;
    const api = await new Promise((resolve, reject) => {
      const abort = () => reject(new DOMException('Aborted', 'AbortError'));
      signal.addEventListener('abort', abort, { once: true });
      loadTurnstile().then(resolve, reject).finally(() => signal.removeEventListener('abort', abort));
      if (signal.aborted) abort();
    });
    if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
    return new Promise((resolve, reject) => {
      let timer;
      let widgetID = null;
      let finished = false;
      const finish = (error, token) => {
        if (finished) return;
        finished = true;
        clearTimeout(timer); signal.removeEventListener('abort', abort);
        if (widgetID !== null) { try { api.remove(widgetID); } catch {} }
        $('[data-ai-verification]').hidden = true;
        error ? reject(error) : resolve(token);
      };
      const abort = () => finish(new DOMException('Aborted', 'AbortError'));
      signal.addEventListener('abort', abort, { once: true });
      timer = setTimeout(() => finish(new Error('verification_failed')), 120000);
      $('[data-ai-verification]').hidden = false;
      const options = { sitekey: root.dataset.siteKey, action: 'docs_ask', theme: 'auto', appearance: 'interaction-only', execution: 'execute', callback: token => finish(null, token), 'error-callback': () => finish(new Error('verification_failed')), 'expired-callback': () => finish(new Error('verification_failed')) };
      try {
        widgetID = api.render($('[data-ai-turnstile]'), options);
        if (finished) api.remove(widgetID);
        else api.execute(widgetID);
      } catch { finish(new Error('verification_failed')); }
    });
  }

  function endpointURL() {
    try {
      const endpoint = new URL(root.dataset.endpoint);
      if (endpoint.protocol !== 'https:' && !(endpoint.protocol === 'http:' && ['127.0.0.1', 'localhost'].includes(endpoint.hostname))) return null;
      if (endpoint.username || endpoint.password || endpoint.search || endpoint.hash) return null;
      return new URL('v1/chat', endpoint.href.endsWith('/') ? endpoint : `${endpoint.href}/`).href;
    } catch { return null; }
  }

  async function submit(question = input.value.trim(), mode = selectedMode, retryTurn = null) {
    if (controller || !question.trim()) return;
    if (question.length > 2000) { setNotice('Please keep your question within 2000 characters.'); return; }
    if (/\bsk-[a-zA-Z0-9_-]{20,}\b/.test(question)) { setNotice('Please do not share API keys, passwords, or other sensitive information.'); return; }
    const endpoint = endpointURL();
    if (!endpoint) { setNotice('Ask AI is not connected yet. Please try again later.'); return; }
    const id = ++operation;
    const abort = new AbortController(); controller = abort;
    const history = requestHistory(retryTurn ? turns.slice(0, turns.indexOf(retryTurn)) : turns);
    const turn = retryTurn || createTurn(question, mode);
    if (!retryTurn) turns.push(turn);
    else { turn.answer = ''; turn.sources = []; turn.complete = false; turn.element.querySelector('[data-ai-answer]').replaceChildren(); turn.element.querySelector('[data-ai-actions]').replaceChildren(); turn.element.querySelector('[data-ai-sources]').replaceChildren(); }
    const status = turn.element.querySelector('[data-ai-status]');
    status.hidden = false; status.classList.remove('is-error'); status.textContent = mode === 'find' ? 'Searching the docs…' : 'Thinking…';
    input.value = ''; setMode('answer'); updateInput(); updateBusy(true); setNotice();
    pinnedToBottom = true; followScroll(true);
    let done = false;
    try {
      const token = await verificationToken(abort.signal);
      const response = await fetch(endpoint, { method: 'POST', credentials: 'omit', headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' }, body: JSON.stringify({ question, mode, currentPage: currentPage(), history, clientId: clientID(), ...(token ? { turnstileToken: token } : {}) }), signal: abort.signal });
      if (!response.ok) {
        const retry = Number(response.headers.get('retry-after'));
        const errors = { 429: `Too many requests. ${Number.isFinite(retry) && retry > 0 ? `Try again in ${Math.ceil(retry)} seconds.` : 'Please try again later.'}`, 403: 'Verification failed. Please try again.', 503: 'Ask AI is temporarily unavailable. Please try again later.' };
        throw new Error(errors[response.status] || 'Unable to connect. Check your network and try again.');
      }
      if (!response.headers.get('content-type')?.includes('text/event-stream')) throw new Error('The response could not be read. Please try again.');
      for await (const event of readEvents(response.body)) {
        if (id !== operation) return;
        if (event.event === 'sources') {
          turn.sources = validatedSources(event.data.sources, docsBase);
          // Navigation is decided by the server's document-intent/relevance
          // checks, never by parsing a model-generated command or arbitrary URL.
          turn.navigation = validatedNavigation(event.data.navigation, turn.sources, docsBase);
          // Safe, data-only diagnostics for browser QA; never store a query,
          // provider response, token or credential in diagnostic attributes.
          const retrieval = event.data.retrieval;
          if (['hybrid', 'lexical'].includes(retrieval?.method)) turn.element.dataset.retrievalMethod = retrieval.method;
          if (typeof retrieval?.semanticExecuted === 'boolean') turn.element.dataset.semanticExecuted = String(retrieval.semanticExecuted);
          if (retrieval?.unavailable) setNotice('Document search is temporarily unavailable. Chat is still available.');
          else if (event.data.indexStale) setNotice('Using a cached documentation index. Recent changes may not appear yet.');
          status.textContent = mode === 'find' ? 'Finding pages…' : 'Writing an answer…';
        } else if (event.event === 'delta' && typeof event.data.text === 'string') {
          turn.answer += event.data.text;
          if (turn.answer.length > 64000) throw new Error('The response is too long and has been stopped.');
          status.hidden = true; renderAnswer(turn);
        } else if (event.event === 'done') {
          done = true; turn.complete = true; turn.truncated = Boolean(event.data.truncated);
        } else if (event.event === 'error') { throw new Error('The response was interrupted. Please try again.'); }
      }
      if (!done) throw new Error('The connection was interrupted. This response may be incomplete.');
      if (!turn.answer && !turn.sources.length) throw new Error('The response was empty. Please try again.');
      status.hidden = true;
      renderAnswer(turn, true); renderSources(turn); renderActions(turn);
      if (turn.truncated) { status.textContent = 'Response limit reached. Ask a follow-up to continue.'; status.hidden = false; }
      save(); announce(turn.sources.length ? 'Response complete. Document sources are available.' : 'Response complete.');
      if (turn.navigation && turn.navigation !== currentPage()) {
        const target = prepareDocumentNavigation(turn.navigation);
        if (target) location.assign(target);
      }
    } catch (error) {
      if (id !== operation) return;
      renderAnswer(turn, true); renderSources(turn);
      status.hidden = false;
      if (abort.signal.aborted) { status.textContent = 'Response stopped'; turn.complete = Boolean(turn.answer); renderActions(turn); announce('Response stopped'); }
      else {
        status.classList.add('is-error');
        status.textContent = error.message?.startsWith('verification') ? 'Verification is unavailable. Check your network and try again.' : error.message?.includes('fetch') || error.name === 'TypeError' ? 'Unable to connect. Check your network and try again.' : ['invalid_event','incomplete_event','stream_too_large','empty_stream'].includes(error.message) ? 'The response could not be read. Please try again.' : error.message;
        const retry = document.createElement('button'); retry.type = 'button'; retry.className = 'tm-ai__source-toggle'; retry.textContent = 'Retry'; retry.dataset.aiRetry = '';
        turn.element.querySelector('[data-ai-actions]').append(retry);
        announce(status.textContent);
      }
    } finally {
      if (id === operation) { controller = null; updateBusy(false); followScroll(); }
    }
  }

  function reset() {
    operation++; controller?.abort(); controller = null;
    if (renderFrame) cancelAnimationFrame(renderFrame); renderFrame = null;
    restoredScrollPending = false;
    pinnedToBottom = true;
    turns = []; messages.replaceChildren(); welcome.hidden = false;
    input.value = ''; setMode('answer'); setNotice(); updateInput(); updateBusy(false);
    try { sessionStorage.removeItem(storageKey); } catch {}
    input.focus({ preventScroll: true }); announce('New chat started');
  }

  async function copyText(text, button, label) {
    try { await navigator.clipboard.writeText(text); button.title = 'Copied'; announce('Copied'); if (label) button.textContent = 'Copied'; }
    catch { announce('Copy unavailable. Select the text and copy it manually.'); }
    setTimeout(() => { if (button.isConnected) { button.title = label || 'Copy response'; if (label) button.textContent = label; } }, 1800);
  }

  root.addEventListener('click', event => {
    if (event.target.closest('[data-ai-open]')) return open();
    if (event.target.closest('[data-ai-close]')) return close();
    if (event.target.closest('[data-ai-new]')) return reset();
    if (event.target.closest('[data-ai-clear-mode]')) { setMode('answer'); input.focus(); return; }
    const prompt = event.target.closest('[data-ai-prompt]');
    if (prompt) { setMode(prompt.dataset.aiPrompt); input.focus(); return; }
    if (event.target.closest('[data-ai-bottom]')) { pinnedToBottom = true; followScroll(true); return; }
    const turn = turns.find(item => item.element.contains(event.target));
    if (!turn) return;
    const code = event.target.closest('[data-ai-copy-code]');
    if (code) return copyText(code.parentElement.querySelector('code')?.textContent || '', code, 'Copy code');
    const copy = event.target.closest('[data-ai-copy]'); if (copy) return copyText(turn.answer, copy);
    const vote = event.target.closest('[data-ai-vote]');
    if (vote) { turn.vote = turn.vote === vote.dataset.aiVote ? null : vote.dataset.aiVote; renderActions(turn); save(); announce('Feedback saved on this device only'); return; }
    if (event.target.closest('[data-ai-retry]')) submit(turn.question, turn.mode, turn);
  });

  // Material's instant navigation sees a normal same-origin anchor and keeps this
  // root mounted. Modified clicks retain native new-tab behavior.
  root.addEventListener('click', event => {
    const link = event.target.closest('a[data-ai-document]');
    if (!link || event.button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    announce(`Opening ${link.textContent.trim().slice(0, 120)}`);
    // MkDocs may fall back to a full navigation for newly inserted anchors.
    // Restore the dock only on that exact target, never on unrelated visits.
    prepareDocumentNavigation(link.dataset.aiDocument);
  });
  send.addEventListener('click', event => { if (controller) { event.preventDefault(); controller.abort(); input.focus({ preventScroll: true }); } });
  composer.addEventListener('submit', event => { event.preventDefault(); submit(); });
  input.addEventListener('input', updateInput);
  input.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!controller) submit(); }
  });
  panel.addEventListener('keydown', event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); } });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && root.classList.contains('is-open') && !document.querySelector('#__search:checked') && !document.querySelector('.tm-mobile-nav[open]')) {
      event.preventDefault(); close();
    }
  });
  thread.addEventListener('scroll', () => { pinnedToBottom = atBottom(); $('[data-ai-bottom]').hidden = pinnedToBottom; }, { passive: true });
  window.visualViewport?.addEventListener('resize', () => { document.documentElement.style.setProperty('--tm-ai-visual-height', `${visualViewport.height}px`); scheduleFollowScroll(); }, { passive: true });
  // Late font/math layout and drawer/composer resizing can change the bottom
  // after history was rendered. Keep following only while the reader is pinned.
  const scrollLayoutObserver = new ResizeObserver(scheduleFollowScroll);
  scrollLayoutObserver.observe(messages);
  scrollLayoutObserver.observe(thread);
  window.addEventListener('pagehide', save);
  document.addEventListener('tm:ask-ai', event => { open(); if (typeof event.detail?.question === 'string') { input.value = event.detail.question.slice(0, 2000); updateInput(); } });
  const refreshContext = () => {
    root.dataset.currentPage = currentPage() || '';
    document.body.classList.toggle('tm-ai-open', root.classList.contains('is-open'));
    try {
      const navigation = JSON.parse(sessionStorage.getItem(navigationStorageKey) || 'null');
      if (navigation?.target === location.href) {
        sessionStorage.removeItem(navigationStorageKey);
        if (navigation.expires > Date.now() && !mobile.matches && !root.classList.contains('is-open')) open();
      } else if (navigation?.expires <= Date.now()) sessionStorage.removeItem(navigationStorageKey);
    } catch {}
  };
  if (typeof document$ !== 'undefined') document$.subscribe(refreshContext);
  else refreshContext();

  try {
    const saved = JSON.parse(sessionStorage.getItem(storageKey) || 'null');
    if (saved?.expires > Date.now() && Array.isArray(saved.turns) && saved.turns.length <= 6) {
      turns = saved.turns.flatMap(turn => {
        if (typeof turn.question !== 'string' || typeof turn.answer !== 'string' || turn.question.length > 2000 || turn.answer.length > 16000) return [];
        return [createTurn(turn.question, ['answer','find','guide'].includes(turn.mode) ? turn.mode : 'answer', { answer: turn.answer, sources: validatedSources(turn.sources, docsBase), complete: true, vote: ['up','down'].includes(turn.vote) ? turn.vote : null })];
      });
    } else sessionStorage.removeItem(storageKey);
  } catch { try { sessionStorage.removeItem(storageKey); } catch {} }
  updateInput();
  restoredScrollPending = turns.length > 0;
  // Desktop document navigation may have opened the dock before rehydration.
  // Otherwise consume this one-time restoration on the first explicit open.
  positionRestoredConversation();
  scheduleFollowScroll();
}

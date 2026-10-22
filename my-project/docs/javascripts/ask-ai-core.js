/** Shared, dependency-free protocol checks. No model-generated URL is trusted. */
export function trustedDocURL(value, baseURL) {
  try {
    if (typeof value !== 'string' || value.length > 2048 || /[\\\u0000-\u001f]/.test(value)) return null;
    const base = new URL(baseURL);
    const url = new URL(value, base);
    if (url.protocol !== 'https:' || url.origin !== base.origin || url.username || url.password || url.search) return null;
    const prefix = base.pathname.endsWith('/') ? base.pathname : `${base.pathname}/`;
    if (!url.pathname.startsWith(prefix)) return null;
    let decoded = url.pathname;
    for (let attempt = 0; attempt < 3; attempt++) {
      const next = decodeURIComponent(decoded);
      if (next === decoded) break;
      decoded = next;
    }
    if (decoded.includes('\\') || decoded.split('/').some(segment => segment === '..' || segment === '.')) return null;
    return url.href;
  } catch { return null; }
}

export function validatedSources(value, baseURL) {
  if (!Array.isArray(value)) return [];
  const ids = new Set();
  return value.slice(0, 12).flatMap(source => {
    const url = trustedDocURL(source?.url, baseURL);
    const id = Number(source?.id);
    if (!url || !Number.isInteger(id) || id < 1 || id > 99 || ids.has(id)) return [];
    ids.add(id);
    return [{ id, url, title: String(source.title || 'Documentation').slice(0, 200), excerpt: String(source.excerpt || '').slice(0, 500) }];
  });
}

export function courseImagePath(value, baseURL, inventory = {}) {
  const fallback = 'assets/start/fallback/signals-gradient.webp';
  const url = trustedDocURL(value, baseURL);
  if (!url) return fallback;
  const path = new URL(url).pathname.slice(new URL(baseURL).pathname.length);
  let course;
  try { course = decodeURIComponent(path.split('/')[0]).toLowerCase().replace(/[^a-z0-9]/g, ''); }
  catch { return fallback; }
  const image = inventory[course];
  return typeof image === 'string' && image.startsWith('assets/start/courses/') && !image.includes('..') && /^assets\/start\/courses\/[^/\\]+\.(?:webp|png|jpe?g|avif)$/i.test(image) ? image : fallback;
}

export function sourcePageTitle(title) {
  return String(title || '').split(' · ')[0].trim() || 'Documentation';
}

/** Only the server's retrieved sources may be automatic navigation targets. */
export function validatedNavigation(value, sources, baseURL) {
  const url = trustedDocURL(value?.url, baseURL);
  return url && Array.isArray(sources) && sources.some(source => source.url === url) ? url : null;
}

/** Data-only SSE, including partial UTF-8 code points and CRLF boundaries. */
export async function* readEvents(body) {
  if (!body?.getReader) throw new Error('empty_stream');
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      pending += done ? decoder.decode() : decoder.decode(value, { stream: true });
      if (pending.length > 262144) throw new Error('stream_too_large');
      let boundary;
      while ((boundary = /\r?\n\r?\n/.exec(pending))) {
        const block = pending.slice(0, boundary.index);
        pending = pending.slice(boundary.index + boundary[0].length);
        let event = 'message';
        const data = [];
        for (const line of block.split(/\r?\n/)) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
        }
        if (data.length) {
          try { yield { event, data: JSON.parse(data.join('\n')) }; }
          catch { throw new Error('invalid_event'); }
        }
      }
      if (done) {
        if (pending.trim() && !pending.trim().startsWith(':')) throw new Error('incomplete_event');
        return;
      }
    }
  } finally { try { await reader.cancel(); } catch {} reader.releaseLock(); }
}

export function requestHistory(turns) {
  return turns.filter(turn => turn.complete && turn.answer).slice(-2).flatMap(turn => [
    { role: 'user', content: turn.question.slice(0, 2000) },
    { role: 'assistant', content: turn.answer.slice(0, 2000) },
  ]);
}

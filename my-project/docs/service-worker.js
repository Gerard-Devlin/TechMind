const CACHE_VERSION = "techmind-v2";
const PAGE_CACHE = `${CACHE_VERSION}-pages`;
const ASSET_CACHE = `${CACHE_VERSION}-assets`;
const SITE_ROOT = new URL("./", self.registration.scope);

const PRECACHE_URLS = [
  new URL("./", SITE_ROOT).href,
  new URL("manifest.webmanifest", SITE_ROOT).href,
  new URL("assets/images/pwa/icon-192.png", SITE_ROOT).href,
  new URL("assets/images/pwa/icon-512.png", SITE_ROOT).href,
  new URL("assets/images/pwa/icon-maskable-512.png", SITE_ROOT).href
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(ASSET_CACHE)
      .then(cache => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", event => {
  const activeCaches = new Set([PAGE_CACHE, ASSET_CACHE]);
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(key => key.startsWith("techmind-") && !activeCaches.has(key))
          .map(key => caches.delete(key))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", event => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin || !url.pathname.startsWith(SITE_ROOT.pathname)) return;

  // Media elements rely on Range requests. Caching a partial 206 response as a
  // normal asset can leave subsequent reloads with an unusable, black video.
  if (request.headers.has("range") || request.destination === "video" || request.destination === "audio") return;

  if (request.mode === "navigate") {
    event.respondWith(networkFirstPage(request));
    return;
  }

  event.respondWith(staleWhileRevalidateAsset(request, event));
});

async function networkFirstPage(request) {
  const cache = await caches.open(PAGE_CACHE);

  try {
    const response = await fetch(request);
    if (response.ok) await cache.put(request, response.clone());
    return response;
  } catch {
    return (await cache.match(request)) ||
      (await caches.match(new URL("./", SITE_ROOT).href)) ||
      Response.error();
  }
}

async function fetchAndCacheAsset(request) {
  const response = await fetch(request);
  if (response.ok) {
    const cache = await caches.open(ASSET_CACHE);
    await cache.put(request, response.clone());
  }
  return response;
}

function staleWhileRevalidateAsset(request, event) {
  const update = fetchAndCacheAsset(request);
  event.waitUntil(update.catch(() => undefined));
  return caches.match(request).then(cached => cached || update);
}

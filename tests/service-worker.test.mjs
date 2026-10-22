import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../my-project/docs/service-worker.js', import.meta.url), 'utf8');
const base = 'https://gerard-devlin.github.io/TechMind/';

function route(path, { mode = 'cors', destination = '', method = 'GET', headers = {} } = {}) {
  const listeners = {};
  const context = vm.createContext({ URL, self: {
    registration: { scope: base }, location: { origin: new URL(base).origin },
    addEventListener(name, callback) { listeners[name] = callback; },
  } });
  vm.runInContext(source, context);
  vm.runInContext('networkFirstPage = () => "network-first-page"; staleWhileRevalidateAsset = () => "cached-asset";', context);
  let result = null;
  listeners.fetch({ request: { url: new URL(path, base).href, mode, destination, method, headers: new Headers(headers) },
    respondWith(value) { result = value; }, waitUntil() {} });
  return result;
}

test('navigation and MkDocs instant HTML requests get fresh pages before cached fallback', () => {
  assert.equal(route('start/', { mode: 'navigate' }), 'network-first-page');
  assert.equal(route('COA/chapter/'), 'network-first-page');
  assert.equal(route('start/index.html?from=nav'), 'network-first-page');
  assert.equal(route('extensionless', { headers: { accept: 'text/html,*/*' } }), 'network-first-page');
  assert.equal(route('extensionless', { destination: 'document' }), 'network-first-page');
});

test('static assets stay cached, while media ranges, external requests and POST are not intercepted', () => {
  assert.equal(route('stylesheets/ask-ai.css?v=6'), 'cached-asset');
  assert.equal(route('javascripts/ask-ai.js?v=6'), 'cached-asset');
  assert.equal(route('video.mp4', { destination: 'video' }), null);
  assert.equal(route('audio.mp3', { headers: { range: 'bytes=0-99' } }), null);
  assert.equal(route('https://techmind-ai.devl1n.workers.dev/v1/chat', { method: 'POST' }), null);
  assert.equal(route('https://outside.example/image.png'), null);
});

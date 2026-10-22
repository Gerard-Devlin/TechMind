# Ask AI replication

Reference: <https://developers.openai.com/>. Inspected on 2026-10-23 in a fresh, isolated, unauthenticated browser context. Scope is the bottom-right Docs agent, not the surrounding developer portal.

## Implementation route

The original embeds an `openai-chatkit` custom element with a hosted cross-origin iframe. Its API is the ChatKit thread/event protocol, not a provider-independent chat-completions endpoint. TechMind therefore reconstructs the measured shell and interaction states as maintainable native components, with the site's own Cloudflare SSE backend. It does not mirror the hosted iframe, reuse its public domain key, or send TechMind questions to OpenAI's reference service.

The original header SVGs, panel dimensions, typography hierarchy, spacing and responsive behavior were inspected directly. OpenAI Sans Regular/Medium already existed locally; Semibold was also saved locally for the outer shell. Subsequent inspection of the actual answer iframe established that its thread, welcome state, card and composer use the system UI font stack, not OpenAI Sans; those inner components now use the measured stack. Markdown uses pinned, vendored marked 18.1.1 and DOMPurify 3.4.17, with the original licenses retained beside them. The only new third-party runtime dependency is Cloudflare Turnstile.

## Measured baseline

| Part | Reference |
| --- | --- |
| Launcher | Bottom/right 20px; 44px tall pill; 14px/20px medium text |
| Desktop panel | 440px wide, full viewport height, left hairline, no corner radius |
| Mobile panel | Below 768px; bottom drawer; `min(78dvh, 640px)` high; top corners 16px |
| Header | 64px; horizontal padding 16px; title 14px semibold; 32px icon buttons with 16px icons |
| Dark surface | Panel `#212121`, composer `#303030` |
| Empty state | Bottom-aligned greeting, three 48px action rows, compact rounded composer |
| Greeting | 28px/34px semibold |
| Composer | About 58px tall in the single-line state; 36px send/stop control |
| Surrounding page | Desktop content and header narrow by panel width; mobile content shifts up by drawer height |
| Generation | User bubble; search/writing status; streaming answer; send arrow becomes stop square |
| Documents | Explicit page request may navigate; broader requests show candidate text and a Related page card with Open page |
| Related page card | Original ChatKit `Card` defaults to `size=sm`: width 360px/max-width 100%, 16px padding, 20px radius; left-aligned, not stretched to the full message width |
| Card title | Original `Title` defaults to `md` / medium: 20px, 26px line-height, weight 500 |
| Open page | Secondary/outline treatment: transparent fill, 1px inset outline, pill corners; default large control height 36px |
| Card footer | Small muted documentation badge on its own line, followed by a full-width outlined Open page control; not a horizontal justify-between row |
| Actual related image | Explicit server widget ratio `1600 / 755`, not its natural 1200×630 ratio or the component's default 16/9; inner image 326×153.828px at 360px card width, cover/center, 8px clipping radius |
| Actual card structure | Outer column gap12: label → image → content → button. Content column gap8: title → excerpt → badge. No per-child guessed margins |
| Inner font | `ui-sans-serif, -apple-system, system-ui, "Segoe UI", "Noto Sans", Helvetica, Arial`, then emoji fallbacks |

In the reference, “Find a page” opened the Models documentation while retaining the conversation. A separate request for several audio candidates produced a list and a Related page card. These were observed, not inferred from a static screenshot.

## Coverage and evidence

Screenshots were captured and actually inspected inline. The screenshot tool refused a file destination inside the workspace, so no saved PNG files are claimed. Agent-isolated DevTools page IDs during this work were reference **2** and local **3**; they are session-specific, not durable evidence URLs.

| State / workflow | Viewport | Local result | Evidence / limits |
| --- | --- | --- | --- |
| Launcher → empty dock | 1440×1000, dark | Inspected | Measured local panel 440px, header 64px, composer 58px; no horizontal overflow |
| Empty drawer | 390×844, dark | Inspected | Local rect x0/y204/w390/h640, matching reference; top corners and compact input compared visually |
| Tablet dock | 834×1112, light | Inspected | Local panel x394/w440, header on remaining 394px; no horizontal overflow |
| Theme switch | 1440×1000 | Passed | Real palette toggle; panel/header/thread/composer rectangles were identical before and after; surface changed to white |
| Candidate cards | 390×844, dark | Passed with mock SSE | Related page, local thumbnail, source title/excerpt and Open page shown; fixture is not a production RAG answer |
| Citation → actual section | 390×844 | Passed | Click reached the real COA pipeline page at `#_4`; target ID existed; mobile drawer closed; conversation survived navigation |
| Answer Markdown / code / math | Phone and desktop | Passed with mock SSE | Strong text, fenced Python code and KaTeX formula displayed; restored conversation re-typeset math |
| Unsafe model content | Phone | Passed with mock SSE | Foreign Markdown link became plain text; raw HTML image/script elements were absent; only trusted document citation remained clickable |
| 503 / Retry | 1440×1000 | Passed with mock HTTP | English error; Retry reused the existing turn instead of adding a duplicate |
| Follow-up context | 1440×1000 | Passed with mock SSE | Request contained the real current section URL, bounded history and a session UUID; no key or URL query token |
| Stop during verification | 1440×1000 | Passed | Stop immediately cleared busy state and removed the Turnstile iframe; no Worker request had been sent |
| New chat | Phone | Passed | Messages removed, empty state returned and session conversation storage cleared |
| Clipboard and feedback controls | Implemented | Partial validation | Copy/code-copy and local-only vote state are wired; clipboard write policy and final live-answer action loop need final browser confirmation |
| Desktop cross-document restoration | Desktop | Passed | A real Open page click reached the local DSA document; its desktop dock remained open and the exact-target navigation flag was consumed |
| Real Worker → provider stream | Local production configuration | Passed | The real “RAW 和 load-use 冲突有什么区别？” request returned a complete DeepSeek answer and two verified section citations. This was a lexical-retrieval request, not a vector-search result |
| Natural greeting | Local production configuration | Passed | After the human completed Turnstile, “你好” returned “你好！有什么想了解的？”. The exact small-talk path bypasses document retrieval and model generation |
| Bare-topic automatic navigation | Real local UI → deployed Worker | Passed | From the Compilers syntax-analysis page, submitting just “网络流” automatically opened the actual PC/4 网络流 document. The phone drawer closed to expose the destination; no Open page click was used |
| Scroll-to-latest hover | Dark and Light, isolated UI | Passed | Actual pointer hover retained opaque `rgb(48,48,48)` / `rgb(243,243,243)` backgrounds. Both screenshots were viewed inline |

Pure tests: `node --test tests/ask-ai-core.test.mjs tests/ask-ai-math.test.mjs` — **10 passed**. They cover confined document URLs including encoded traversal, source-ID filtering, generated course-image matching, short page titles, source-confined automatic navigation, fragmented UTF-8/CRLF SSE, malformed/incomplete events, bounded history, math delimiter preservation and code/raw-HTML handling. The two Python artwork-inventory / safe-attribute tests also passed via `.venv/Scripts/python.exe -m unittest discover -s tests -p test_ask_ai_assets.py`. Syntax checks passed. The mock UI pass had no console errors; the real Turnstile challenge emitted Cloudflare's own diagnostic warnings.

### Answer-card precision follow-up

After the first implementation, the parent agent observed the user's actual original answer: its Related page card was narrower and its Open page control was outlined, whereas the initial TechMind card stretched to the available message width and used a solid button. This discrepancy was corrected against public original runtime code, without submitting another model request or changing the user's browser session.

Original public resources inspected:

- `https://cdn.platform.openai.com/assets/ck1/index-CA-tk7t4.css`: `.jzduf[data-size=sm]{width:360px}`, card padding/radius tokens, `.lkCln[data-variant=outline]` transparent/inset-border rules, title/font and automatic child-spacing tokens.
- `https://cdn.platform.openai.com/assets/ck1/index-Bg-G8goj.js`: `Card` defaults to small, `Title` defaults to medium-size/medium-weight, secondary buttons map to outline and default to large controls.

Initial local CSS verification used a clearly isolated synthetic card fixture, not a claimed real RAG answer. At 834px viewport / 440px dock, the card measured **360px wide**, **16px padding**, **20px radius**, with a **36px-high transparent Open page** and a 1px inset outline. This established those dimensions, but did not validate the supplied image widget's explicit props; the later real-card comparison found its image ratio and gaps were still wrong. The card keeps the reference's spare space to its right. At 390px viewport it contracts to the available message width without horizontal overflow. Light card background is white; dark remains `#303030`. The card is a noninteractive container with an `h3` title and a separate Open page navigation control, not a whole-card link. A Ctrl-click dispatch was observed as not prevented by the app and left the dock/navigation restoration state unchanged. A subsequent real normal click reached the local DSA page and kept the desktop dock open. These screenshots were viewed inline; the earlier screenshot-file limitation still applies.

The parent agent's follow-up observation also confirmed that the documentation badge and full-width Open page are stacked, not arranged in a horizontal footer row; this structure was corrected. Card headings now show only the first ` · `-separated page-title segment, with the full section breadcrumb retained in the title tooltip/accessibility label and the actual section hash retained in the navigation target.

The earlier static-fixture pass used the source function and CSS in the agent's isolated local page, without a model request. At 1440×1000 / Light, the card was 360px wide, its local DSA artwork and full-width Open page were both 326px wide, and the latter remained 36px high with a transparent background. The gray documentation badge occupied its own 20px-high row. At 390×844 / Dark, the settled drawer was at y=204 with height 640, the card contracted to 356px, and the image/button contracted to 322px with no horizontal overflow. The heading remained 20px/26px at weight 500; the local image decoded successfully. Both screenshots were actually viewed inline. These controlled layout checks did not establish image/caption fidelity, and are not a claim that the synthetic fixture came from the production RAG service.

The generic `Image` widget defaults to 16/9, but that is **not** the ratio of the actual Related page image. A subsequent read-only measurement inside the original live answer iframe found explicit inline `width:100%;aspect-ratio:1600 / 755` on the image wrapper, producing 326×153.828px. Its natural image is 1200×630, confirming the wrapper explicitly overrides both natural and default ratios. The implementation now follows this real widget: wrapper/image radius8, overflow hidden, no border/shadow/pseudo-element, cover/center image. The actual surrounding column uses gap12, with an inner title/excerpt/badge column using gap8. The badge is 12px/18px at weight500, height20 and horizontal padding6.65. The Open page outline is the original `::before` inset 1px shadow (white25% in Dark); the card's own 1px border is white12% in Dark. This real-instance evidence supersedes the earlier default-token inference.

The v4 isolated-fixture check verified the implemented geometry against that real-card measurement: at 1440×1000 the card was 360px, image 326×153.828px, button 326×36px, image radius8, card radius20 and border `1px solid rgba(255,255,255,.12)`. Measured label→image, image→content and content→button gaps were each exactly12px; title→excerpt and excerpt→badge were exactly8px. The button's `::before` had the measured white25% inset1px shadow. At 390×844, the card was356px and the image322×151.9375px, with the same gaps and no horizontal overflow. Both screenshots were viewed inline. Computed thread and composer fonts matched the measured system UI stack. No model request or user-owned tab refresh was used for this pass.

The v5 real-answer check found a separate behavioral discrepancy: ordinary answers hid their sources in an invented Sources accordion. The reference displays one Related page directly, so all modes now render a single visible related card. Every section citation remains available in the answer and find mode retains its textual candidate list. The real RAW/load-use answer was reloaded and inspected with its matching local COA image, compact page heading, excerpt, badge and Open page. An additional live DOM read corrected the dark title/button color to `#fff` and the documentation badge text to `#dcdcdc`; its padding is `0 6.65px` with centered alignment. The scroll-to-latest button previously inherited the generic translucent icon hover; its specific hover now uses the opaque field surface instead, preventing message text from showing through. Narrow composers explicitly suppress horizontal scrolling.

The v6 actual-component mock-SSE checks repeated the card comparison at 1440×1000, 834×1112 and 390×844 with 1303 characters of Chinese text. Desktop/tablet cards measured360, images326×153.828; phone cards356, images322×151.9375. Radius20/padding16/outergap12/innergap8/buttonheight36 remained unchanged. Page and thread had zero horizontal overflow. The long phone placeholder has greater intrinsic scrollWidth but is clipped by `overflow-x:hidden`, without leaking outside its composer. Esc closed the panel and made it inert. These screenshots were viewed inline; the tool again refused the requested workspace screenshot-file destination, so no PNG file is claimed.

An old Service Worker was also found serving the MkDocs instant-navigation HTML through the asset stale-while-revalidate path. The worker is now `techmind-v3`: navigation, HTML Accept/document requests and slash/HTML routes all use network-first with the existing offline fallback. Static assets retain caching and media Range requests remain untouched. Two VM routing tests passed. Registration uses `updateViaCache:none`, and the chat scripts/styles have v6 cache-busting URLs. The production backend has no AI/Vectorize bindings or embedding calls; its retired `techmind-docs` cloud index was explicitly deleted at the user's request. The final lexical backend passed50 Node tests, a real workerd/SQLite smoke test and private-repository CI.

### Restored conversation scrolling

The v7 frontend fixes refresh reopening a saved conversation at its first turn. A six-turn browser fixture reproduced `scrollTop=0` and a6051px bottom gap at1440×1000 before the fix; afterward the same fixture reached `scrollTop=6051` and bottom gap0. Restoration is consumed once, after history rendering or on the first open, including desktop document navigation that opens before rehydration. A coalesced animation-frame callback and ResizeObserver follow late font/content and thread-size changes only while the reader remains pinned to the latest message.

Isolated real-browser checks passed for desktop1440×1000 and phone390×844: initial restored opening at bottom; desktop auto-open before hydration; delayed content/font reflow while pinned; manual scrolling to300px followed by late content, close/reopen or drawer resizing retaining300px; the latest-message button returning to gap0; and New chat clearing all turns without an old callback reviving them. Desktop and phone screenshots were viewed inline. Browser fixtures used only local saved messages, with verification disabled in the isolated test context; no real model request or user's conversation was modified. The available browser tools were used because the local Python environment has no Playwright installation.

## Intentional product differences

- OpenAI/DevDay content becomes TechMind documentation and course context. `Ask AI`, `Docs agent`, `Find a page` and `Build a custom guide` retain the reference's control hierarchy; service statuses are English and model answers follow the user's language.
- Candidate descriptions come from server-selected document excerpts, not editorial strings. Related page thumbnails use the matching local course artwork from a build-generated asset inventory. Unknown course slugs use the existing unbranded `signals-gradient.webp` abstract fallback; no hardcoded Codex image or remote asset link is used.
- Bare topics such as “网络流”, high-confidence technical questions and explicit document requests may automatically open the selected page or section after the response completes. Ambiguous, broad, multi-topic or negated requests retain candidates rather than blindly opening the first hit. Automatic targets must exactly match a validated server source; arbitrary model-generated URLs cannot trigger navigation. Every model-generated link must also match that source table.
- On phones, choosing a document closes the drawer so the destination is readable. Desktop keeps the dock, including a guarded full-navigation fallback.
- There are no fake uploads, fake server-side thumbs-up submissions, or inactive external actions. Votes stay in the current browser session only; copy uses the browser clipboard API.
- Conversation history is bounded and stored only in sessionStorage with a 30-minute expiry. The server does not receive a full page or entire site. A separate random 30-minute `clientId` isolates answer caches and is not authentication.
- No hidden model reasoning is displayed or requested. Search/writing states are actual transport states, not fabricated thought steps.
- The original's hosted custom-guide publishing service is not reproduced. Guide mode generates an answer grounded in real course pages and links, without claiming to create a separately hosted document.
- The Light/Dark surfaces switch in the same DOM, so there is no cross-iframe theme reinitialization or position change.

## Source ownership

- `material/partials/ask-ai.html`: public configuration and accessible shell.
- `docs/stylesheets/ask-ai.css`: scoped visual tokens, measured geometry and responsive layout.
- `docs/javascripts/ask-ai.js`: UI, safe Markdown, Turnstile lifecycle, SSE conversation and navigation.
- `docs/javascripts/ask-ai-core.js`: URL/source validation, stream parser and bounded history.
- `hooks/ask_ai_assets.py`: automatically scans local course image filenames at build time; the source URL's first course path segment is normalized and matched to that inventory, without another manually maintained course registry.
- `docs/assets/vendor/ask-ai/`: local rendering libraries and licenses; metadata uses `.txt`, not indexed Markdown.
- `tests/ask-ai-*.test.mjs`: focused regression tests outside the documentation corpus.

`extra.ask_ai.endpoint` and `extra.ask_ai.site_key` in MkDocs are public configuration. DeepSeek and Turnstile secret keys never belong in these files. A final MkDocs build/deploy is still required to ship the current source tree; no frontend publication is claimed by this document.

# Start banner and mobile navigation

Inspected on October 22, 2026. The banner reference is https://developers.openai.com/; the mobile navigation reference is https://openai.com/zh-Hans-CN/, following the user's later correction.

## Implementation and intentional differences

- The banner reuses publicly delivered browser components, CSS, eye artwork, and local fonts. These are compiled browser bundles, not the original application source.
- The banner chooses one of six short three-word headlines on each full page load: Make ideas 「real」; Think. Build. 「Repeat」; Stay curious. 「Explore」; Think beyond 「limits」; Build what 「matters」; Learn. Try. 「Grow」. A session-storage guard avoids consecutive repeats. The selected line remains stable during theme changes and instant navigation. The final word has corner quotes and no trailing full stop.
- Highlight ranges are supplied per headline, not hardcoded character positions. The accessible heading and non-animated fallback use the same selected copy. Description: "Notes on code, computer systems, and machine intelligence." Actions: Browse subjects and View source.
- Banner text, surface, and actions follow the selected site theme. Both themes use identical padding, width, and reserved title geometry. The banner creates an isolated stacking context.
- Courses use local static images. The two former course MP4 files were removed; course catalog entries and chapter counts remain derived from the configured navigation.
- Desktop featured courses, All subjects, and footer use the hero's 1152px shell with 40px inner gutters. Mobile gutters are unchanged.
- Following the later tablet correction, 704–1023px uses a 16:9 lead card above exactly three square cards in a three-column row, with 32px gutters, 16px column gaps and 48px lead-to-row spacing. Below 704px the portrait/stacked phone layout remains unchanged; desktop retains its sticky lead and four-card rail.
- The compact header now hides the entire desktop search trigger through the 1220px navigation breakpoint, instead of only giving its wrapper zero width. This removes the floating Ctrl K badge at 960–1219px. Opening the search palette does not add a margin to the compact theme control.
- Mobile navigation now follows OpenAI.com's large root list, full-screen opaque panel, small back control, muted section heading, and large child links. It no longer uses the developer site's pills, suggested chips, search field, or fixed footer.
- The mobile header has a left-aligned TechMind brand and adjacent 40px controls on the right: theme, search, and sidebar. The panel header has search and sidebar-close controls.
- Both compact brand marks match the footer's OpenAI Sans family, 600 weight, and tight tracking, scaled to 20px. The hidden desktop search wrapper reserves no mobile width.
- Root course labels reuse the navbar's generated short names. All child panels and chapter links are generated recursively from the full navigation, including courses pruned from Material's inactive sidebar.
- The root menu footer contains All subjects and GitHub. The requested theme-switch sentence was removed; theme switching remains in the site header.
- Native dialog behavior handles top-layer stacking, focus containment, and Escape. Panel navigation preserves scroll positions and returns focus to the course button.
- The menu remains available through Material's 1220px desktop breakpoint so tablet article navigation stays accessible. OpenAI.com's desktop breakpoint differs.
- The menu uses locally downloaded OpenAI Sans v4 Latin fonts. Chinese chapter text keeps the site's system-font fallback.
- Search and sidebar icons are inline stroke SVGs reconstructed from the visible controls, since direct original icon downloads and the browser asset export were unavailable. Back/external-arrow paths are reused from the inspected DOM. Asset provenance is in docs/assets/navigation/sources.json.
- The bottom-right AI assistant / DS Flash integration remains a TODO; no API integration was added.

## Reference evidence

The original mobile menu was inspected in an existing signed-out Chrome tab at 389 × 644. Root and child screenshots were viewed, and the actual DOM and computed styles were read. Key source values: 54px mobile / 64px tablet header; 24px mobile / 32px tablet panel inset; 16px root-list spacing; fluid 32–48px headings; 13px child-section headings; 48px footer separation plus 32px top padding. The source's Research submenu and Home back action were inspected with keyboard activation.

An isolated browser context encountered Cloudflare; it was not bypassed. The user's already accessible public Chrome page supplied the original UI evidence.

## Preview and validation

The preview serves the generated .codex-tmp/site output at http://127.0.0.1:8794/start/.

```powershell
.\.venv\Scripts\python.exe -m mkdocs build -f my-project\mkdocs.yml -d .codex-tmp\site
.\.venv\Scripts\python.exe -m http.server 8794 --bind 127.0.0.1 --directory .codex-tmp\site
```

Browser checks use Chrome emulation, not physical devices or other browser engines. Existing site-wide math/font CDN dependencies are unchanged. New banner and navigation assets load locally. No deployment was performed.

### Confirmed checks

- Production build succeeds; both new JavaScript entries pass Node syntax checks. Existing navigation/plugin warnings remain unchanged.
- At 1440px, hero, featured courses, catalog, and footer share content edges x=184 and x=1256. No horizontal page overflow.
- Theme switching preserves the measured hero, title, featured-section, and catalog boxes at 320px, 390px, 768px, and 1440px. The chosen headline stays unchanged.
- All six headline alternatives were rendered at 320px. After adding corner quotes, glyph widths were checked and the mobile font reduced to 21cqw; the longest final word, 「matters」, fits its 140px title area.
- Consecutive normal refreshes produced different headlines. Corner quotes appear in the rendered title; there is no trailing full stop.
- The phone header's three controls occupy adjoining 40px boxes. Header/panel brand family, weight, and tracking follow the footer. Light and dark outline icons were inspected; their shape fill is none.
- The modal has an opaque surface, contains focus, and locks the page scroll. C expands to 11 chapters; the Back action returns to the root. A chapter click opens the correct document and unlocks the page. Tablet menu access and the desktop sidebar were checked.
- Menu search opens the real search palette and focuses its input. NumPy returns 42 matching documents, including the local NumPy chapter; Escape closes search.
- Static-asset/catalog audit confirms 26 static raster images, no animated course assets, 10 subjects, and chapter counts matching YAML. No console warnings or errors in the final Start page flow.
- At 834 × 1112, the tablet lead image measures 770 × 433.125px; the three lower cards are each 246px wide with equal y=662.484px top edges and 16px gaps. The fourth rail item is hidden. The screenshot was inspected and page overflow is zero.
- At 1024px, theme/search/menu occupy adjoining 40px boxes, the hidden desktop trigger occupies zero space, and opening/closing search preserves those coordinates. The phone remains one lead plus three vertically stacked cards.

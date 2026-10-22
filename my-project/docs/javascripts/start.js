/* Mount the adapted reference components inside the MkDocs page lifecycle. */
(() => {
  const runtime = new URL("../assets/start/devday/runtime/", import.meta.url);
  let revision = 0;
  let roots = [];
  let components;
  const phrases = [
    { text: "Make ideas 「real」", green: "ideas", purple: "real" },
    { text: "Think. Build. 「Repeat」", green: "Think", purple: "Build" },
    { text: "Stay curious. 「Explore」", green: "curious", purple: "Explore" },
    { text: "Think beyond 「limits」", green: "beyond", purple: "limits" },
    { text: "Build what 「matters」", green: "Build", purple: "matters" },
    { text: "Learn. Try. 「Grow」", green: "Learn", purple: "Try" }
  ];
  let chosenPhrase;

  function phrase() {
    if (chosenPhrase) return chosenPhrase;
    let last = -1;
    try { last = Number(sessionStorage.getItem("tm-start-phrase") ?? -1); } catch {}
    const choices = phrases.map((value, index) => ({ ...value, index })).filter(value => value.index !== last);
    chosenPhrase = choices[Math.floor(Math.random() * choices.length)];
    try { sessionStorage.setItem("tm-start-phrase", String(chosenPhrase.index)); } catch {}
    return chosenPhrase;
  }

  function prepareTitle(banner) {
    const selected = phrase();
    banner.querySelector(".tm-start-sr-only").textContent = selected.text;
    banner.dataset.titlePhrase = String(selected.index);
    const fallback = banner.querySelector(".tm-start-wordmark__fallback");
    fallback.replaceChildren();
    selected.text.split(" ").forEach((word, index) => {
      if (index) fallback.append(document.createTextNode(" "));
      const span = document.createElement("span");
      const token = document.createElement("span");
      token.textContent = word.replace(/[「」]/g, "");
      if (word.includes(selected.green)) token.className = "tm-start-wordmark__green";
      else if (word.includes(selected.purple)) token.className = "tm-start-wordmark__purple";
      if (word.startsWith("「")) span.append("「", token, "」");
      else span.append(token);
      fallback.append(span);
    });
    const colorRules = ["green", "purple"].map(color => ({
      scope: "range",
      start: selected.text.indexOf(selected[color]),
      end: selected.text.indexOf(selected[color]) + selected[color].length,
      dotColor: `var(--tm-devday-${color})`,
      outlineColor: "currentColor",
      textColor: `var(--tm-devday-${color})`
    }));
    return { text: selected.text, colorRules };
  }

  async function mount() {
    const current = ++revision;
    roots.forEach(root => root.unmount());
    roots = [];

    const banner = document.querySelector(".tm-start-devday");
    if (!banner) return;
    const titleProps = prepareTitle(banner);

    try {
      components ??= Promise.all([
        import(new URL("TechMindHeroTitle.js?v=20261022-7", runtime)),
        import(new URL("TechMindAgent.js", runtime)),
        import(new URL("client.F-XekeHr.js", runtime)),
        import(new URL("react.CGSMBlYx.js", runtime))
      ]);
      const [title, agent, client, react] = await components;
      if (current !== revision || !banner.isConnected) return;

      const { createElement } = react.t();
      const { createRoot } = client.t();
      const titleRoot = createRoot(banner.querySelector("[data-tm-devday-title]"));
      const agentRoot = createRoot(banner.querySelector("[data-tm-devday-agent]"));
      roots = [titleRoot, agentRoot];
      titleRoot.render(createElement(title.default, titleProps));
      agentRoot.render(createElement(agent.default, { label: "Show another agent" }));
    } catch (error) {
      // The HTML title and local illustration remain visible without the runtime.
      components = undefined;
      console.warn("TechMind DevDay components could not start:", error);
    }
  }

  if (typeof document$ !== "undefined") document$.subscribe(mount);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount, { once: true });
  else mount();
})();

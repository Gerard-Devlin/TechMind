document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "$$",  right: "$$",  display: true },
      { left: "$",   right: "$",   display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true }
    ],
  })

  const mathRoot = body.querySelector(".md-typeset")
  if (!mathRoot) return

  let measureFrame
  const markWideInlineMath = () => {
    cancelAnimationFrame(measureFrame)
    measureFrame = requestAnimationFrame(() => {
      mathRoot.querySelectorAll("span.arithmatex").forEach((formula) => {
        formula.classList.remove("tm-math-overflow")

        const content = formula.querySelector(".katex, mjx-container") || formula
        const formulaBox = content.getBoundingClientRect()
        const line = formula.closest("li, p, td, th, dd") || mathRoot
        const lineBox = line.getBoundingClientRect()
        const availableWidth = lineBox.right - formula.getBoundingClientRect().left

        // Ignore tiny rounding/font-rendering differences. Those formulas still
        // fit visually and should remain ordinary inline math.
        if (formulaBox.width > availableWidth + 16) {
          formula.classList.add("tm-math-overflow")
        }
      })
    })
  }

  markWideInlineMath()

  const observer = new MutationObserver(markWideInlineMath)
  observer.observe(mathRoot, { childList: true, subtree: true })

  const resizeObserver = new ResizeObserver(markWideInlineMath)
  resizeObserver.observe(mathRoot)
})

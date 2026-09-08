// Prompt additions for pasted-PDF passages. Kept in sync with bench/prompts/pdf-hint-short.txt
// and bench/prompts/pdf-fewshot2.json, which the tier benchmark runs with.
export const PDF_HINT = "The text was copied from a PDF: superscripts/subscripts are flattened (x2 = x^2, kB = k_B),\nfractions and limits are split across lines (a lone symbol on its own line is a numerator,\ndenominator or limit of the nearby formula), zero-width characters and line wraps are noise.\nReturn the passage as plain text in the original order with each formula converted: \\( \\) inline, \\[ \\] for a formula on its own line. Never wrap prose in \\text{}, never use enumerate or equation environments, never drop or reorder sentences, never solve anything.";

export const PDF_FEWSHOT = [
  {
    "user": "A damped oscillator satisfies m x¨ + c x˙ + kx = 0. For light damping the solution decays as x(t) = A e\n−γt\ncos(ωt + φ), where γ =\nc\n2m\n​\n\nand the period is T = 2π/ω. The\nsheet lists ω0 =\n√\nk/m for the undamped case.",
    "assistant": "A damped oscillator satisfies \\( m\\ddot{x} + c\\dot{x} + kx = 0 \\). For light damping the solution decays as \\( x(t) = A e^{-\\gamma t} \\cos(\\omega t + \\varphi) \\), where \\( \\gamma = \\frac{c}{2m} \\) and the period is \\( T = 2\\pi/\\omega \\). The sheet lists \\( \\omega_0 = \\sqrt{k/m} \\) for the undamped case."
  },
  {
    "user": "The partial sums converge quickly. In particular\n∞\n\n1\nπ2\n∑ 2 =\nk\n8\nk=1, odd\n​\n\nand the ratio test uses lim (ak+1)\nk→∞ ak\n. Both facts appear in chapter 4.",
    "assistant": "The partial sums converge quickly. In particular \\[ \\sum_{k=1,\\, \\text{odd}}^{\\infty} \\frac{1}{k^2} = \\frac{\\pi^2}{8} \\] and the ratio test uses \\( \\lim_{k\\to\\infty} \\frac{a_{k+1}}{a_k} \\). Both facts appear in chapter 4."
  }
];

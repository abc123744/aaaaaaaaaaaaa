# Numerical Experiment Revision Notes

This repository note records the improvements made to the numerical study after submission. It is not proposed Appendix G text; the camera-ready-style replacement is [`APPENDIX_G_REPLACEMENT.md`](APPENDIX_G_REPLACEMENT.md).

## What changed

- **Simulation scale:** each tested $q$ now uses $N=10^{10}$ trajectories in each of the add and remove directions, aggregated from 20 independent shards per direction.
- **Direct bounded estimation:** $\delta(\varepsilon)$ is estimated through bounded hockey-stick expectations rather than generalized-Pareto tail extrapolation.
- **Pessimistic reporting:** the plotted orange curve uses the simultaneous upper curve plus the $\beta=10^{-9}$ release term.
- **Simultaneous uncertainty:** the MC confidence allocation covers the full fixed grid of 700,000 candidate $\varepsilon$ points for each of five $q$ values and both adjacency directions; 1,500 points per curve are exported only for display.

## What remains unchanged

The adaptive rule, the five parameter configurations, the filter budget, and the theoretical CLT are unchanged. The new computation replaces the submitted numerical estimation and presentation, not the theoretical results.

## Rebuttal-specific scope

The replacement numerical study instantiates one AdaGrad-inspired history-dependent rule. It illustrates finite-regime behavior and does not estimate the supremum over admissible adaptive rules. The simultaneous band is gridwise rather than continuous in $\varepsilon$, and the comparison is with the zero-slack $G_{\sqrt{2B}}$ reference rather than the theorem's finite-regime approximate-GDP guarantee with parameter $\sqrt{2B}+O(\cdot)$ and slack $\Delta$.

The U-shaped discrepancy is consistent with the two asymptotic regimes, but the experiment does not empirically prove the theorem's uniform asymptotic statement or provide a deterministic practical privacy certificate.

## Where to verify the revision

- [`README.md`](README.md): concise experiment summary.
- [`APPENDIX_G_REPLACEMENT.md`](APPENDIX_G_REPLACEMENT.md): proposed scientific presentation.
- [`TABLES.md`](TABLES.md): browser-viewable numerical tables.
- [`data/coverage_audit.json`](data/coverage_audit.json): machine-readable confidence construction.
- [`code/audit_coverage.py`](code/audit_coverage.py): reproducible confidence audit.
- [`figures/`](figures/): main replacement figure and one figure for each $q$.
- [`tables.tex`](tables.tex): generated LaTeX source for the tables.

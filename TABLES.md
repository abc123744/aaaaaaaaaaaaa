# Numerical Tables

These are browser-viewable copies of the three paper-facing tables generated in [`tables.tex`](tables.tex).

## Empirical directional CDF discrepancies

| $q$ | remove $\widehat\Delta_{\mathrm{rem}}$ | add $\widehat\Delta_{\mathrm{add}}$ | final $\widehat\Delta$ | GDP parameter $\sqrt{2B}$ |
|---:|---:|---:|---:|---:|
| 0.01 | 0.003533 | 0.003377 | 0.003533 | 0.1775 |
| 0.1 | 0.01228 | 0.01450 | 0.01450 | 0.5612 |
| 0.199 | 0.01862 | 0.02584 | 0.02584 | 0.7897 |
| 0.801 | 0.006391 | 0.01461 | 0.01461 | 1.416 |
| 0.95 | 0.001889 | 0.004735 | 0.004735 | 1.454 |

The $\widehat\Delta$ values are empirical estimates from the 700,000-bin CDF histogram rather than confidence upper bounds.

## Moments

| $q$ | $B$ | Gaussian mean | Gaussian variance | $P$ mean | $P$ variance | $Q$ mean | $Q$ variance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.01575 | $\pm0.01575$ | 0.03150 | 0.01570 | 0.03150 | -0.01565 | 0.03121 |
| 0.1 | 0.1575 | $\pm0.1575$ | 0.3150 | 0.1532 | 0.3142 | -0.1494 | 0.2913 |
| 0.199 | 0.3118 | $\pm0.3118$ | 0.6236 | 0.2967 | 0.6187 | -0.2846 | 0.5464 |
| 0.801 | 1.003 | $\pm1.003$ | 2.005 | 1.006 | 2.087 | -0.9693 | 1.866 |
| 0.95 | 1.058 | $\pm1.058$ | 2.115 | 1.058 | 2.139 | -1.046 | 2.066 |

## Halting behavior

| $q$ | halt under $P$ | halt under $Q$ | mean halting step under $P$, among halted | mean halting step under $Q$, among halted | mean halting epoch under $P$, among halted | mean halting epoch under $Q$, among halted |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 100.00% | 100.00% | 1111.2 | 1111.2 | 11.11 | 11.11 |
| 0.1 | 98.87% | 98.80% | 113.1 | 113.1 | 11.31 | 11.31 |
| 0.199 | 99.24% | 99.13% | 58.0 | 58.0 | 11.53 | 11.54 |
| 0.801 | 98.61% | 98.06% | 14.8 | 14.8 | 11.85 | 11.88 |
| 0.95 | 99.98% | 99.97% | 11.8 | 11.9 | 11.22 | 11.29 |

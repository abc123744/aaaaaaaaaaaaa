# Appendix G: Numerical Validation

This appendix reports the numerical validation of the approximate GDP filter.

## G.1 Experimental setup

For a remove pair, let $P=\mathcal M(S)$ and $Q=\mathcal M(S^-)$, and write

$$
L(y)=\ln\frac{P(y)}{Q(y)}.
$$

We simulate the fully adaptive stopped PLRV in both adjacency directions: removal compares $P$ to $Q$, while addition compares $Q$ to $P$. Every step uses independent Poisson subsampling. The principal experiment uses the AdaGrad-inspired rule

$$
\sigma_t=\min\!\left(\sigma_0,\max\!\left(\sigma_{\min},\frac{\sigma_0}{\sqrt{\sum_{i=1}^{t-1}y_i^2}}\right)\right),\qquad C_t=\frac{C_0}{\sqrt{\sum_{i=1}^{t-1}y_i^2}}.
$$

with

$$
\sigma_0=8,\qquad \sigma_{\min}=2,\qquad C_0=1.
$$

The first step uses $(\sigma_0,C_0)$. At later steps, the filter caps the normalized sensitivity when necessary to exhaust but not exceed the remaining budget. Thus the noise, clipping, privacy-loss increments, and stopping time depend on the previously released outputs.

We use the following five configurations:

| $q$ | regime $\tilde q$ | composition length $T$ | $B/\mathrm{Budg}(\tilde q,q,\sigma_0,1)$ | $B$ | $\sqrt{2B}$ |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 0 | 1200 | 20000 | 0.01575 | 0.1775 |
| 0.1 | 0 | 120 | 2000 | 0.1575 | 0.5612 |
| 0.199 | 0 | 65 | 1000 | 0.3118 | 0.7897 |
| 0.801 | 1 | 20 | 200 | 1.003 | 1.416 |
| 0.95 | 1 | 20 | 150 | 1.058 | 1.454 |

For each $q$ and each of the add and remove directions, we merge 20 independent shards of $5\times10^8$ trajectories, giving $N=10^{10}$ trajectories per direction:

- **Remove:** compare $P$ to $Q$, sample $y\sim P$, and use $L(y)=\ln(P(y)/Q(y))$.
- **Add:** compare $Q$ to $P$, sample $y\sim Q$, and use $-L(y)=\ln(Q(y)/P(y))$.

The cumulative PLRV includes the increments through the binding step at which the filter stops.

## G.2 Direct privacy-curve estimation

We estimate the two directed privacy curves using the bounded hockey-stick identities

$$
\widehat\delta_{\mathrm{rem}}(\varepsilon)=\frac1N\sum_{i=1}^N\left[1-\mathrm e^{\varepsilon-L_i}\right]_+,
$$

and

$$
\widehat\delta_{\mathrm{add}}(\varepsilon)=\frac1N\sum_{i=1}^N\left[1-\mathrm e^{\varepsilon+L_i}\right]_+,\qquad y_i\sim Q.
$$

Each summand is bounded in $[0,1]$, eliminating subtraction between independently estimated tails and removing the need for a parametric tail model.

The final add/remove privacy curve is

$$
\widehat\delta(\varepsilon)=\max\!\left\{\widehat\delta_{\mathrm{rem}}(\varepsilon),\widehat\delta_{\mathrm{add}}(\varepsilon)\right\}.
$$

The plotted lower and upper curves use empirical-Bernstein bounds whose failure probability is allocated over both sides of the full fixed grid of $5\times2\times700{,}000=7{,}000{,}000$ candidate $\varepsilon$ points, with total failure probability $\beta=10^{-9}$. We export only 1,500 points per curve for display; since the simultaneous event already covers the full deterministic grid, this selection does not change the guarantee. The orange curve is the EVR-style pessimistic release $U_\beta(\varepsilon)+\beta$, where $U_\beta$ is the final add/remove simultaneous upper curve.

## G.3 Main-rule results

### Empirical directional CDF discrepancies

| $q$ | remove $\widehat\Delta_{\mathrm{rem}}$ | add $\widehat\Delta_{\mathrm{add}}$ | final $\widehat\Delta$ | GDP parameter $\sqrt{2B}$ |
|---:|---:|---:|---:|---:|
| 0.01 | 0.003533 | 0.003377 | 0.003533 | 0.1775 |
| 0.1 | 0.01228 | 0.01450 | 0.01450 | 0.5612 |
| 0.199 | 0.01862 | 0.02584 | 0.02584 | 0.7897 |
| 0.801 | 0.006391 | 0.01461 | 0.01461 | 1.416 |
| 0.95 | 0.001889 | 0.004735 | 0.004735 | 1.454 |

These $\widehat\Delta$ values are empirical estimates from the 700,000-bin CDF histogram, rather than confidence upper bounds; the result JSONs separately record the histogram-grid and DKW slacks.

### Moments

| $q$ | $B$ | Gaussian mean | Gaussian variance | $P$ mean | $P$ variance | $Q$ mean | $Q$ variance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.01575 | $\pm0.01575$ | 0.03150 | 0.01570 | 0.03150 | -0.01565 | 0.03121 |
| 0.1 | 0.1575 | $\pm0.1575$ | 0.3150 | 0.1532 | 0.3142 | -0.1494 | 0.2913 |
| 0.199 | 0.3118 | $\pm0.3118$ | 0.6236 | 0.2967 | 0.6187 | -0.2846 | 0.5464 |
| 0.801 | 1.003 | $\pm1.003$ | 2.005 | 1.006 | 2.087 | -0.9693 | 1.866 |
| 0.95 | 1.058 | $\pm1.058$ | 2.115 | 1.058 | 2.139 | -1.046 | 2.066 |

### Halting behavior

| $q$ | halt under $P$ | halt under $Q$ | mean halting step under $P$, among halted | mean halting step under $Q$, among halted | mean halting epoch under $P$, among halted | mean halting epoch under $Q$, among halted |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 100.00% | 100.00% | 1111.2 | 1111.2 | 11.11 | 11.11 |
| 0.1 | 98.87% | 98.80% | 113.1 | 113.1 | 11.31 | 11.31 |
| 0.199 | 99.24% | 99.13% | 58.0 | 58.0 | 11.53 | 11.54 |
| 0.801 | 98.61% | 98.06% | 14.8 | 14.8 | 11.85 | 11.88 |
| 0.95 | 99.98% | 99.97% | 11.8 | 11.9 | 11.22 | 11.29 |

## G.4 Figures for all sampling rates

### $q=0.01$

[Open the numerical-validation figure for $q=0.01$ (PDF).](figures/CLT_validation_0.01.pdf)

### $q=0.1$

[Open the numerical-validation figure for $q=0.1$ (PDF).](figures/CLT_validation_0.1.pdf)

### $q=0.199$

[Open the numerical-validation figure for $q=0.199$ (PDF).](figures/CLT_validation_0.199.pdf)

### $q=0.801$

[Open the numerical-validation figure for $q=0.801$ (PDF).](figures/CLT_validation_0.801.pdf)

### $q=0.95$

[Open the numerical-validation figure for $q=0.95$ (PDF).](figures/CLT_validation_0.95.pdf)

Panels (a)-(b) show the remove direction under $y\sim P$ and its empirical absolute CDF discrepancy from the $\mathcal N(B,2B)$ reference. Panels (c)-(d) show the add direction under $y\sim Q$ and its empirical discrepancy from $\mathcal N(-B,2B)$. Panel (e) compares the EVR-style pessimistic upper add/remove DP curve with $G_{\sqrt{2B}}$ and the matched $(1,B)$ RDP conversion; the shaded region is the $1-10^{-9}$ MC confidence band.

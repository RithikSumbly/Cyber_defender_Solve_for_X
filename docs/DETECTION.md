# SENTRY detection design

How SENTRY detects model extraction, why each mechanism was chosen over the common alternatives, and the research directions that extend it to production-scale fleets.

## Contents

1. [What the detector observes](#1-what-the-detector-observes)
2. [The five signals](#2-the-five-signals)
3. [Per-key baselines and fusion](#3-per-key-baselines-and-fusion)
4. [Leg 1: anytime-valid per-key sequential test](#4-leg-1-anytime-valid-per-key-sequential-test)
5. [Leg 2: campaign correlation](#5-leg-2-campaign-correlation)
6. [Leg 3: fleet specialization gap](#6-leg-3-fleet-specialization-gap)
7. [Response: alerts and throttling](#7-response-alerts-and-throttling)
8. [The extraction odometer](#8-the-extraction-odometer)
9. [Design choices and the alternatives they outperform](#9-design-choices-and-the-alternatives-they-outperform)
10. [Validation method](#10-validation-method)
11. [Research directions for production-scale deployments](#11-research-directions-for-production-scale-deployments)
12. [References](#12-references)

---

## 1. What the detector observes

Model extraction cannot be seen in a single request. Each query is a well-formed, authenticated, paid-for call. What gives the attack away is the shape of the traffic over time and across keys: how fast a key asks, how broadly it spreads across the classifier's classes, how close successive inputs sit to each other, how often answers land near the decision boundary, and how much new input space a key keeps reaching.

SENTRY therefore watches a stream, not a request. For every answered query the inference API appends one compact record to a log:

| Field | Purpose |
|---|---|
| timestamp | rates, windows, time to alert |
| API key | per-key baselines and cross-key correlation |
| predicted class | class diversity |
| top-class confidence | boundary probing |
| 16-dimensional embedding | a fixed projection of the model's internal representation, used for distance, coverage and campaign similarity |

Raw customer inputs are never stored. The detector runs as a separate operating-system process that reads this log and evaluates a 5-second trailing window for every active key once per second. Because the API never imports the detector, the inference path carries zero detection overhead (p99 `/predict` latency 26.2 ms over 56,065 requests) and keeps serving if the detector stops.

## 2. The five signals

The brief names four example signals. SENTRY implements all four and adds a fifth.

| Signal | Definition over the window | Extraction behaviour it exposes |
|---|---|---|
| Query rate per key | queries divided by window length | high-volume harvesting |
| Input diversity / entropy | Shannon entropy of the predicted-class histogram, normalised to [0, 1] | boundary mapping across many classes |
| Distance between consecutive queries | mean L2 distance between successive embeddings (lower is more suspicious) | perturbation sweeps around a seed input |
| Share of low-confidence answers | fraction of answers with top-class confidence below 0.55 | deliberate probing of the decision boundary |
| Embedding coverage growth | fraction of the window's queries that land in a feature-space cell the key has never visited | systematic mapping of the input space |

Each signal targets a different part of the extraction playbook. Legitimate users settle into a stable pattern (their coverage growth falls toward zero as they revisit familiar inputs), while an extractor keeps pushing into new territory, near the boundary, across many classes.

## 3. Per-key baselines and fusion

Every signal is converted into a one-sided z-score against the **key's own** calibrated baseline:

```
z_s = max(0, d_s * (x_s - mu_key,s) / sigma_key,s)       d_s = -1 for distance, +1 otherwise
fused score = sum over signals of z_s
```

Only deviations in the suspicious direction count. The signal with the largest z-score is the one named in the alert, so an operator sees *why* a key fired, not just that it fired. A key with no calibrated history is scored against the population baseline until it has one.

Per-key baselines are what let SENTRY leave the nightly batch partner alone. It runs at 20 queries per second, above a static rate limit and nearly three times the per-key rate of the split-key attackers, and it is judged against its own normal rate instead of a global limit.

## 4. Leg 1: anytime-valid per-key sequential test

A detector that re-checks a fixed threshold on every one-second step is running a new hypothesis test every second. Across a long session those checks accumulate chances of a spurious crossing: a 99th-percentile threshold checked 130 times has up to a 1 - 0.99^130 (about 73%) chance of at least one false crossing. SENTRY's per-key leg uses e-values instead, which are built for continuous monitoring (Vovk & Wang, 2021).

For every window and every signal:

1. **Conformal p-value.** The observed value is ranked against the key's benign calibration samples (pooled samples when the key has none): `p = (1 + #calibration values at least as extreme) / (1 + n)`.
2. **Calibrator.** `e = kappa * p^(kappa - 1)` with `kappa = 0.4`. This function integrates to 1 over [0, 1], so under benign behaviour the e-value has expectation at most 1: a fair bet against the null.
3. **Combination.** The five e-values are averaged with fixed weights. An average of e-values is itself an e-value under *arbitrary* dependence between them (Vovk & Wang, 2021), so correlated signals combine without a second calibration stage.
4. **Accumulation.** A Shiryaev-Roberts e-detector accumulates evidence across windows, `R_t = (1 + R_(t-1)) * E_t`, and raises an alert when `R_t >= 1/alpha` with `alpha = 0.001`, a threshold of 1,000.

The alarm threshold is derived from theory, not tuned on traffic. On identical traffic, a per-window z-score threshold fires on **15.0%** of normal-user trials, while the e-value leg fires on **0.0%** (0 of 60). This leg detects the high-volume flood (A1) in 15 of 15 trials with a mean of 9.9 s, and the boundary-probing sweep from the sealed image pool (A3) in 15 of 15 trials with a mean of 9.0 s.

## 5. Leg 2: campaign correlation

An attacker can spread one extraction campaign across many API keys so that no single key looks unusual. Published per-account detectors are defeated this way: Kabir et al. (2026) measured PRADA's F1 falling from 63.2% to 0.0% against 400 round-robin Sybil clients.

SENTRY evaluates keys together:

1. For every active key, compute the centroid of its window embeddings.
2. Link any two keys whose centroids have cosine similarity of at least 0.80, and take connected components with union-find, so campaigns of any size are grouped (not only pairs).
3. Sum the members' fused scores and compare with the campaign threshold, the 99th percentile over 93 benign clusters (6.99).
4. Fire when the same group stays above threshold for 3 consecutive polls, then throttle every member.

The five-key split (A6), matched to the normal class mix and run at an individually unremarkable 7 queries per second per key, is detected in 15 of 15 trials with a mean of 5.0 s. The per-key e-value leg alone scores 0% on A6, which is exactly why the legs are combined.

## 6. Leg 3: fleet specialization gap

A more capable attacker controls what each key asks for. Giving each key a disjoint slice of the class space (two classes per key across five keys) produces keys that are individually narrow and that do not resemble one another. Pooled together, those keys still cover the whole class space. SENTRY measures that signature directly, without needing any link between the keys:

```
gap = H(class histogram of all eligible keys pooled) - mean over keys of H(class histogram of that key)
```

Entropies are normalised to [0, 1]. Keys with at least 5 queries in the window are eligible, so a key that has only just started does not distort the statistic. The leg fires when the gap reaches 0.307, the 99th percentile over 349 benign fleet windows, on 3 consecutive polls.

The statistic is identity-blind: it needs no account metadata and no similarity between keys. The adaptive split (A7), written with full knowledge of the detector and designed to evade similarity-based campaign grouping, is detected in 15 of 15 trials with a mean of 5.1 s.

## 7. Response: alerts and throttling

When any leg fires, SENTRY responds through all three channels the brief lists:

- **Console log.** One `[ALERT]` line naming the leg, the top signal, the score and the threshold.
- **Dashboard row.** An incident ranked by severity (multi-key campaign and fleet alerts are critical, single-key alerts are medium) with its evidence: per-signal z-scores, member scores for a campaign, or pooled against per-key entropy for a fleet alert. Operators can acknowledge, resolve or mark each incident.
- **Automatic throttling.** The detector calls the API's throttle endpoint for each flagged key, which enforces a 1-second minimum interval for 120 seconds. Faster requests receive HTTP 429. Answers stay correct and every other key is unaffected.

A 15-second cooldown per key or group keeps a sustained attack from flooding the incident feed.

## 8. The extraction odometer

Detection answers "is this key attacking?". The odometer answers "how much of the model has been exposed, whoever asked?". It accumulates a per-query leakage estimate over the pooled stream of every key:

```
leak(query) = (1 - confidence) * 1 / (1 + prior visits to the query's feature-space cell)
```

Uncertain answers reveal more about where the decision boundary lies, and repeated regions reveal less each time. Because the stream is pooled across keys, splitting a campaign across accounts does not lower the reading. Across 60 attack episodes the odometer tracks the fidelity of the attacker's real substitute clone with Pearson **r = 0.86**.

## 9. Design choices and the alternatives they outperform

All comparisons below use identical logged traffic, 15 trials per cell.

| Detector | A1 flood | A3 sealed sweep | A6 five-key split | A7 adaptive split | Normal-user FPR |
|---|---|---|---|---|---|
| Static per-key rate limit | 100% | 0% | 0% | 0% | 25.0% |
| Distance-only (PRADA-style) | 0% | 100% | 0% | 0% | 16.7% |
| MMD-style fixed window | 0% | 40% | 0% | 0% | 6.7% |
| Per-window z-score threshold | 100% | 100% | 100% | 13% | 15.0% |
| SENTRY leg 1 alone | 100% | 100% | 0% | 0% | 0.0% |
| SENTRY leg 3 alone | 0% | 0% | 0% | 100% | 0.0% |
| **SENTRY, three legs** | **100%** | **100%** | **100%** | **100%** | **0.0%** |

**Per-key baselines instead of a static rate limit.** A flat limit flags every session of the 20 qps batch partner, 25.0% of all normal-user trials, and still misses every attack that is not simply fast. Judging each key against its own history separates a busy customer from an extractor.

**Several signals instead of one.** Distance-only detection, the idea behind PRADA, catches perturbation sweeps and nothing else, and it misfires on ordinary browsing. A fixed-window distribution test picks up part of one attack family. Extraction has several behavioural fingerprints, so SENTRY measures five of them.

**Sequential e-values instead of a threshold re-checked every second.** Continuous monitoring with a fixed threshold turns every second into a fresh chance of a false alarm (15.0% measured). E-values accumulate evidence with a threshold derived from theory (0.0% measured).

**Three independent legs instead of one fused per-key score.** Per-key evidence cannot see a campaign split across keys, similarity grouping cannot see keys that deliberately differ, and the fleet statistic targets exactly that split. Each leg covers a distinct attack structure, and the combination reaches 100% on every family at 0.0% false positives. OR-combining the legs keeps every alert attributable to a named leg and signal.

**Benign-only calibration instead of a supervised attack classifier.** A classifier trained on attack traffic learns the attacks it was shown and needs attack labels, which invites train/test leakage. SENTRY calibrates every threshold on normal traffic alone, then proves generalisation on an attack drawn from a sealed image pool that calibration never touched (A3, 15 of 15).

**Throttling instead of blocking or poisoning answers.** Blocking a key cuts off revenue, and perturbing outputs degrades answers for everyone. A time-limited throttle is reversible, costs an extractor most of its throughput, and leaves every answer correct.

**A separate process instead of in-path middleware.** Running out of process adds no latency to `/predict`, and the service fails open: detection can be restarted or upgraded without touching the model endpoint.

**Compact embeddings instead of raw inputs.** A 16-dimensional projection carries the geometry the signals need while keeping customer data out of the security log.

## 10. Validation method

- **Data ledger.** Four disjoint CIFAR-10 slices: victim training (50,000 train images), calibration (test 0 to 4,999), evaluation (test 5,000 to 7,999) and a sealed attack pool (test 8,000 to 9,999). A static AST test in `make verify` fails if detector or API code imports attack-traffic or clone-training modules.
- **Calibration.** 99th percentile of benign traffic: 1,758 per-key windows, 93 benign campaign clusters, 349 fleet windows.
- **Protocol.** 8 scenarios, 15 trials each, 120 real-time episodes sent over HTTP at real wall-clock rates. Wilson 95% confidence intervals on every rate.
- **Traffic.** Four normal-user profiles (0.5, 2, 1.2 and 20 queries per second) and four attackers (flood, sealed boundary-probing sweep, five-key split, adaptive split).
- **Clones.** For every attack a substitute model is trained on the attacker's own query and label pairs and scored against the victim on a held-out probe set. Attacks are stopped after 13% to 30% of their query budget.
- **Robustness.** `make demo` runs the acceptance test end to end and passes on back-to-back runs. `make verify` covers 14 unexpected-input cases, the fail-open guarantee and the leakage firewall.

Headline result: **60 of 60** attack episodes detected (Wilson 95% CI [94.0%, 100%]) and **0 of 60** normal-user episodes flagged (Wilson 95% CI [0%, 6.0%]).

## 11. Research directions for production-scale deployments

SENTRY's architecture exposes four extension points: the per-query log, the fused suspicion score, the throttle endpoint, and the key graph built by campaign correlation. Each direction below plugs into one of them and is grounded in published work.

### 11.1 Sequence-level detection that is robust to query strategy

State-of-the-art extractors choose each next query from the answers so far, using uncertainty sampling or boundary-seeking active learning (Chandrasekaran et al., 2020). Their signature is a trajectory: queries keep concentrating near a decision boundary that moves as the substitute improves. VarDetect models the full query sequence of each account in a learned latent space and flags sequences that depart from legitimate usage, independent of how the queries were selected.

**In SENTRY:** a fourth leg that tracks each key's per-query novelty trajectory over the whole session. Benign sessions saturate as users revisit familiar inputs, while an active learner keeps producing new boundary-adjacent queries by design. It is calibrated on long legitimate sessions and validated against a real uncertainty-sampling attacker built on the existing substitute-model harness.

### 11.2 Suspicion-proportional response shaping

Two lines of work degrade a stolen copy without needing a binary detection decision. Prediction poisoning perturbs returned probabilities so that the gradient an attacker trains on points away from the true gradient (Orekondy et al., 2020). Gradient redirection reaches the same goal far more efficiently by steering the attacker's gradient in a chosen direction (Mazeika et al., 2022). Moving-target defence adds the principle of denying the adversary a stable signal to adapt to.

**In SENTRY:** use the fused score as a continuous dial. Low suspicion keeps full-fidelity answers, which covers almost all traffic. Rising suspicion applies gradient-redirected perturbation scaled to the score, and the hard threshold keeps today's throttle, delivered as latency variation instead of an explicit status code. A small set of canary query and response pairs at high suspicion provides forensic evidence if a competing model later reproduces them, following the knowledge-honeypot approach.

### 11.3 Long-horizon campaigns

An extractor willing to spread queries across days produces no anomalous five-second window. Behavioural API-security practice addresses low-and-slow abuse by correlating individually unremarkable activity over long dwell times into one continuously updated risk score.

**In SENTRY:** the extraction odometer already measures cumulative exposure (r = 0.86 against real clone fidelity). The extension adds a per-key burn-rate baseline, fitted exactly like the existing per-key signal baselines, and an exponentially weighted moving average over hours to days, alerting on sustained deviation from a key's own established rate of exposure.

### 11.4 Sybil correlation across thousands of keys

Production fraud detection links accounts through many signals at once: hard links such as shared payment instruments or verified organisations, and soft links such as device, network and timing behaviour. Large-scale systems compress graphs of tens of millions of nodes, embed them (LINE) and cluster them with density-based methods (HDBSCAN) to expose coordinated rings that hard links alone miss. Graph neural networks further strengthen Sybil detection at scale.

**In SENTRY:** campaign correlation becomes a multi-signal key graph. Content similarity stays as one edge weight, and **request-timing correlation** (synchronised launches, matched rates, correlated inter-arrival jitter) is added from timestamps already present in the per-query log. Billing and account links join as hard edges wherever the platform holds them. Connected components give way to community detection (Louvain or HDBSCAN) over a streaming graph store sized for fleets of thousands of keys.

## 12. References

- Tramer, Zhang, Juels, Reiter, Ristenpart. *Stealing Machine Learning Models via Prediction APIs.* USENIX Security 2016.
- Juuti, Szyller, Marchal, Asokan. *PRADA: Protecting Against DNN Model Stealing Attacks.* IEEE EuroS&P 2019.
- Chandrasekaran, Chaudhuri, Giacomelli, Jha, Yan. *Exploring Connections Between Active Learning and Model Extraction.* USENIX Security 2020.
- Vovk, Wang. *E-values: Calibration, Combination and Applications.* Annals of Statistics 49(3), 2021.
- Tang et al. *ModelGuard: Information-Theoretic Defense Against Model Extraction Attacks.* USENIX Security 2024.
- Kabir et al. *AI Model Extraction Attacks: Bypassing Single-Client Assumptions in Defenses.* ICMCIS 2026, arXiv:2606.03381.
- Pal et al. *Stateful Detection of Model Extraction Attacks (VarDetect).* https://arxiv.org/abs/2107.05166
- Orekondy, Schiele, Fritz. *Prediction Poisoning: Towards Defenses Against DNN Model Stealing Attacks.* ICLR 2020.
- Mazeika, Li, Forsyth. *How to Steer Your Adversary: Targeted and Efficient Model Stealing Defenses with Gradient Redirection.* ICML 2022. https://arxiv.org/pdf/2206.14157
- *Toward Proactive, Adaptive Defense: A Survey on Moving Target Defense.* https://arxiv.org/pdf/1909.08092
- *Let Them Steal: Trapping LLM Extraction Attacks with Knowledge Honeypot.* https://arxiv.org/pdf/2606.15810
- *A Systematic Survey of Model Extraction Attacks and Defenses.* https://arxiv.org/pdf/2508.15031
- *Fraud Detection Through Large-Scale Graph Clustering with Heterogeneous Link Transformation.* https://arxiv.org/abs/2512.19061
- *Sybil Detection using Graph Neural Networks.* https://arxiv.org/pdf/2409.08631
- MITRE ATLAS AML.T0024.002 (Extract ML Model), AML.T0048.004 (IP Theft). OWASP Top 10 for LLM Applications 2025, LLM10 (Unbounded Consumption).

# SENTRY: Real-Time Model Extraction Detection

**AI model extraction detection and API query monitoring for inference APIs.**

A proprietary model behind a prediction API can be cloned by sending it large volumes of carefully chosen queries and training a copy on the answers. Every one of those requests is valid, authenticated and paid for, so the theft only exists in the pattern across queries and across API keys. SENTRY reads that pattern in real time, raises an alert that names the signal that fired, and throttles the key while every legitimate customer keeps full service.

- Overview deck: [`SENTRY_ModelExtractionDetection.pdf`](SENTRY_ModelExtractionDetection.pdf)
- Demo video and evidence: https://drive.google.com/drive/folders/1JR1W-Ehb0a3QUHd_aaR6SLzPGiB3LnDf?usp=sharing
- Detection design in depth: [`docs/DETECTION.md`](docs/DETECTION.md)

## Results

Measured over 120 real-time episodes (8 scenarios x 15 trials) sent over HTTP at real wall-clock rates. These are the figures reported in the overview deck, with full per-scenario results in `results/deck_metrics.json`. `make eval` reruns the evaluation end to end and writes a fresh `results/metrics.json`.

| Metric | Result |
|---|---|
| Attack episodes detected (flood, boundary-probing sweep, five-key split, adaptive split) | **60 / 60** (Wilson 95% CI [94.0%, 100%]) |
| False alarms across four normal-user profiles | **0 / 60** (Wilson 95% CI [0%, 6.0%]) |
| Mean time from attack start to alert | 9.9 s, 9.0 s, 5.0 s, 5.1 s (A1, A3, A6, A7) |
| `/predict` latency p50 / p95 / p99 | 11.1 / 18.5 / 26.2 ms over 56,065 requests |
| Share of each attack's query budget spent before the alert | 13% to 30% |
| Static per-key rate limit on the same traffic | 25.0% false alarms, and it misses A3, A6 and A7 |

### Detection matrix

Identical logged traffic for every detector, 15 trials per cell.

| Detector | A1 flood | A3 sealed sweep | A6 five-key split | A7 adaptive split | Normal-user FPR |
|---|---|---|---|---|---|
| Static per-key rate limit | 100% | 0% | 0% | 0% | 25.0% |
| Distance-only (PRADA-style) | 0% | 100% | 0% | 0% | 16.7% |
| MMD-style fixed window | 0% | 40% | 0% | 0% | 6.7% |
| Per-window z-score threshold | 100% | 100% | 100% | 13% | 15.0% |
| SENTRY leg 1 alone (e-value) | 100% | 100% | 0% | 0% | 0.0% |
| SENTRY leg 3 alone (fleet gap) | 0% | 0% | 0% | 100% | 0.0% |
| **SENTRY, three legs** | **100%** | **100%** | **100%** | **100%** | **0.0%** |

## Why a rate limit is not enough

The busiest legitimate customer in this evaluation is a nightly batch integration at 20 queries per second. A static 10 queries per second limit flags every one of its sessions and still misses the three attackers that stay under it, including every key of both split campaigns. SENTRY separates a high-volume customer from a coordinated extraction campaign by judging each key against its own calibrated baseline and by correlating behaviour across keys.

## Architecture

```
Clients (normal users, attackers)
   -> FastAPI inference API: PyTorch CNN image classifier, per-key auth, throttle enforcement
   -> compact per-query log: key, timestamp, predicted class, confidence, 16-d embedding (never raw pixels)
   -> SENTRY detector, a separate OS process: 5 s windows, polled every 1 s
   -> response: console [ALERT] naming the signal, dashboard incident, automatic throttling of the key
```

**Fail-open by design.** The API process never imports the detector. If the detector stops, `/predict` keeps serving. A control that can take down revenue does not survive in production, so SENTRY is built so that it cannot.

**Graded response.** A flagged key is throttled (HTTP 429 under a minimum request interval) rather than blocked or fed wrong answers, so a legitimate customer is never cut off and the model's answers stay correct for everyone.

## How detection works

Five signals are computed per key over each window and scored against that key's own calibrated baseline:

| Signal | What it measures | Extraction behaviour it exposes |
|---|---|---|
| Query rate per key | queries per second | high-volume harvesting |
| Input diversity / entropy | spread of predicted classes | boundary mapping across classes |
| Distance between consecutive queries | embedding distance between successive inputs | perturbation sweeps around a seed |
| Share of low-confidence answers | answers near the decision boundary | deliberate boundary probing |
| Embedding coverage growth | new regions of feature space reached | systematic space mapping |

Three independent detection legs run on those signals, and any one of them raises an alert:

1. **Per-key sequential test.** E-values accumulated by a Shiryaev-Roberts e-detector (Vovk & Wang, *Annals of Statistics*, 2021). The alarm threshold comes from theory (1/alpha, alpha = 0.001) and stays meaningful however long a key is monitored.
2. **Campaign correlation.** Concurrently active keys are grouped by embedding-centroid similarity and their evidence is summed, so a campaign split into individually quiet keys is scored as one campaign.
3. **Fleet specialization gap.** An identity-blind statistic: pooled fleet class entropy minus the mean per-key entropy. Splitting the work into narrow specialists makes the fleet look complete while each key looks unremarkable, and this leg measures exactly that.

The reasoning behind each choice, the alternatives it outperforms, and the research directions for production-scale deployments are in [`docs/DETECTION.md`](docs/DETECTION.md).

## Validation

- **Benign-only calibration.** Every threshold is the 99th percentile of normal traffic: 1,758 per-key windows, 93 benign campaign clusters and 349 fleet windows. No attack data sets a threshold.
- **Data ledger with no train/test leakage.** Victim training, calibration, evaluation and a sealed attack pool are four disjoint CIFAR-10 slices. A static AST test (`tests/test_no_leakage_imports.py`, run by `make verify`) fails if detector or API code imports attack-traffic or clone-training modules.
- **Hard traffic on both sides.** Four normal-user profiles, including the 20 qps batch partner. Four attackers: a high-volume flood, boundary-probing sweeps drawn from the sealed pool, a five-key split matched to the normal class mix, and an adaptive split built with full knowledge of the detector.
- **Real clones.** For every attack a substitute model is trained on the attacker's own query and label pairs and scored against the victim on a held-out probe set. SENTRY's extraction odometer tracks that clone fidelity with Pearson r = 0.86 across 60 attack episodes.
- **Statistics.** Wilson 95% confidence intervals on every rate, three published-style baselines on identical traffic, and a per-window z-score comparison that shows why sequential testing matters (15.0% false alarms against 0.0%).
- **Robustness.** `make demo` passes on back-to-back runs. `make verify` runs 14 unexpected-input cases (forged keys, malformed base64, non-image bytes, empty and non-JSON bodies, a 6 MiB payload, null bytes and path traversal in headers), confirms fail-open by killing the detector mid-run, and runs the leakage firewall.

## Quickstart

```bash
git clone https://github.com/RithikSumbly/Cyber_defender_Solve_for_X.git
cd Cyber_defender_Solve_for_X
make setup      # installs requirements, downloads CIFAR-10 (170 MB), trains the victim model, calibrates on normal traffic
make demo       # acceptance test: normal traffic stays green, the extraction alert fires naming the signal
make verify     # 14 unexpected-input cases, fail-open check, leakage firewall
make eval       # full 120-episode statistical evaluation (about an hour, real traffic)
```

Guided walkthrough of the live system, in one command:

```bash
make walkthrough   # starts the inference API and a detector; open http://127.0.0.1:8081
```

Each step sends real traffic to the running model, from normal customers to four extraction attacks, and reads the result back from the detector's alert log: which leg fired, which signal, which keys, and how many seconds after the attack began. The operations console is embedded beside the steps and switches to the panel that matters as each step runs.

Live operations console, in three terminals:

```bash
make serve                         # inference API on 127.0.0.1:8000
python -m sentry.detect.detector   # detector process
make dashboard                     # operations console on http://127.0.0.1:8080
```

The console shows a KPI strip, live traffic and alert activity, a cost and impact panel, a severity-ranked incident feed with per-signal evidence and triage actions, a key inventory with per-key drill-down, and the live calibrated detection policy. Its Simulate tab launches any normal-user profile or attack against the running system in one click.

## Repository layout

| Path | Contents |
|---|---|
| `sentry/api` | FastAPI inference API, key registry, throttle enforcement |
| `sentry/model` | victim CNN architecture and training |
| `sentry/traffic` | normal-user profiles, attack generators, real-time traffic runner |
| `sentry/detect` | signals, fusion, e-values, campaign clustering, calibration, live detector |
| `sentry/eval` | evaluation harness, baselines, substitute-model fidelity, odometer, statistics |
| `sentry/dashboard` | operations console |
| `sentry/demo` | guided walkthrough of the live system |
| `scripts` | `make demo` and `make verify` |
| `tests` | leakage firewall |
| `results` | calibration, data ledger and evaluation output |
| `docs` | detection design and research directions |

## Threat model

MITRE ATLAS AML.T0024.002 (Extract ML Model) leading to AML.T0048.004 (IP Theft). OWASP LLM10:2025 (Unbounded Consumption). SENTRY deploys beside the API gateway, stores compact embeddings and per-query metadata, and never stores raw customer inputs.

## References

- Tramer et al., *Stealing Machine Learning Models via Prediction APIs*, USENIX Security 2016.
- Juuti et al., *PRADA: Protecting Against DNN Model Stealing Attacks*, IEEE EuroS&P 2019.
- Chandrasekaran et al., *Exploring Connections Between Active Learning and Model Extraction*, USENIX Security 2020.
- Vovk and Wang, *E-values: Calibration, combination and applications*, Annals of Statistics 2021.
- Tang et al., *ModelGuard*, USENIX Security 2024. Output perturbation, complementary to query-pattern detection.
- Kabir et al., ICMCIS 2026, arXiv:2606.03381. Per-account detection under Sybil splitting across 400 clients.


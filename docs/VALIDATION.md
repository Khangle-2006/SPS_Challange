# Publication validation — 15 November 2026

This checks the source release, not a new dataset experiment.

| Check | Outcome |
|---|---|
| Archived source/config/license SHA256 | 882/882 matched original local files |
| All Python syntax | Parsed without importing research workers |
| Staged source bytes / documentation links | All882 source hashes also match Git index bytes; newly authored local documentation links resolve |
| Existing scorer/selection tests | 10/10 PASS on Python3.11, NumPy2.2.6, SciPy1.16.2 |
| New publication smoke test | PASS: synthetic90bpm gives90.087890625 on3/3windows; static RGB0/1accepted, no numeric HR |
| Scorer and selection CLI help | PASS |
| Secret scan | No credential/token/private-key pattern matches in staged source; authentication/SSH/transfer helpers excluded |
| Fresh virtualenv installation | Not verified: local TLS certificate chain prevented pip fetching dependencies; truststore support unavailable in that environment. Existing installed environment used for numerical tests. TLS verification was not disabled. |
| Full research environment / real-data replay | Not rerun for publication; requires authorized private assets and historical runtime configuration |
| Holdout | Not accessed |

Commands executed from the publication root:

```bash
python tools/verify_source.py
python -m unittest test_score_round1
python -m unittest discover -s tools -p 'test_*.py'
python score_round1.py --help
python selection_round1.py --help
```

The portable test covers the POS numerical baseline only, not the selected calibrated BTC estimator. No claim of a validated one-command full reproduction is made.

Archived source has pre-existing trailing whitespace reported by `git diff --check`; it is retained to preserve the original source hashes. `.gitattributes` disables line-ending conversion so a checkout does not silently invalidate these hashes.

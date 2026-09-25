# MergeGuard — Business Entity Resolution

Blocking + matching baseline for the Amazon ML Challenge entity resolution
problem. This is a working starting point, not the final thing — the actual
score depends on how we tune it against the real data (see "Next" below).

## Layout
```
dataset/
  train/train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
  test/test_source1.tsv, test_source2.tsv, test_source3.tsv
src/
  normalize.py   - name/address cleanup (abbreviations, punctuation)
  blocking.py    - TF-IDF cosine candidate generation, per-country + cross-country
  features.py    - pairwise similarity features for the matcher
  train.py       - trains the classifier, tunes threshold for F_0.5
  predict.py     - runs the trained model on test, writes both output files
  utils.py       - IO + our own copy of the F_0.5 scorer
artifacts/        - saved model + threshold (train.py writes here)
output/           - matching_results.tsv + candidate_pairs.tsv (predict.py writes here)
```

Note: `utils/validate_submission.py` is **not** part of this repo — it's
provided by Amazon in the contest's `student_resource/` package. Drop our
`dataset/`, `src/`, `artifacts/`, `output/` folders alongside their
`utils/` folder so the path below resolves.

## Run
```bash
pip install -r requirements.txt
cd src
python train.py --data-dir ../dataset --out-dir ../artifacts
python predict.py --data-dir ../dataset --artifacts-dir ../artifacts --out-dir ../output
python ../utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir ../dataset/test
```

## Next
- Check blocking recall first (train.py prints it per source). If it's not
  near 1.0, fix blocking before touching the model — nothing downstream can
  recover a candidate that never got generated.
- Once there's a trained model, look at the false positives by hand — F_0.5
  punishes them 2x, so these matter more than the misses.
- `MAX_CANDIDATES_PER_SOURCE` in blocking.py trims an unordered set right
  now, which means the trim is arbitrary, not "keep the best." Fix this if
  it's actually triggering on real data — carry similarity scores through
  and cut by score instead.
- Ensembling (second model, different seed/features, average the
  probabilities) is probably worth more than another day of tuning one model.
- Re-run the threshold search any time blocking or features change — the
  old threshold won't be right for a new model.
- difflib is a stdlib stand-in for now; rapidfuzz would likely do a bit
  better on name similarity if we add it to requirements.txt.

## Constraints
- No external data/API/geocoding calls anywhere in here.
- Classifier is LightGBM (falls back to sklearn's HistGradientBoosting if
  LightGBM isn't installed) — both MIT-licensed, nowhere close to 8B params.
- Output format matches the spec: tab-separated, comma-joined IDs, every
  Source-1 test entity present, empty string for singletons.
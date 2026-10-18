"""Chapter 4: several rows per patient, and why the split must keep patients together."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

rng = np.random.default_rng(0)
patients, visits = 100, 10
diagnosis = rng.integers(0, 2, patients)                     # one label per patient
signature = rng.normal(0, 1, (patients, 8))                  # unrelated to the diagnosis
group = np.repeat(np.arange(patients), visits)               # patient of every row
X = signature[group] + rng.normal(0, 0.3, (patients * visits, 8))
X[:, 0] += 1.0 * diagnosis[group]                            # the only real, weak signal
y = diagnosis[group]

model = RandomForestClassifier(n_estimators=200, random_state=0)
rows = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
whole = cross_val_score(model, X, y, groups=group, cv=GroupKFold(5))

if __name__ == "__main__":
    print(f"{patients} patients, {visits} visits each, {len(y)} rows")
    print(f"rows split at random:     accuracy {rows.mean():.3f} +- {rows.std():.3f}")
    print(f"patients kept together:   accuracy {whole.mean():.3f} +- {whole.std():.3f}")

"""The marking script accepts correct results and rejects a wrong one."""


def filled(values):
    rows = "\n".join(f"| `{k}` | {v if isinstance(v, str) else f'{v:.6g}'} |" for k, v in values.items())
    return "| Quantity | Value |\n|---|---|\n" + rows + "\n"


def test_accepts_correct_and_rejects_wrong(load_script, tmp_path):
    verify = load_script("verify_results")
    values = verify.reference_values()
    good = tmp_path / "good.md"
    good.write_text(filled(values))
    assert verify.check(good)
    values["mean_val_accuracy"] += 0.05
    bad = tmp_path / "bad.md"
    bad.write_text(filled(values))
    assert not verify.check(bad)

# Guided exercise

Start in `based-on/`. Its three named functions intentionally raise
`NotImplementedError`, so its acceptance tests fail until completed.
From the companion repository root, run `.venv/bin/python -m pytest`
followed by `chapter-17-vision/applied/based-on`.

Complete each function using the chapter equations and run the acceptance tests.
Compare against `full-code/`, then run `vision_exercise.py` there. The experiment uses
validation data for selection and test data for reporting. The measured record
is `full-code/reference_results.json`; `verify_results.py --rerun` repeats it.
Numbers can vary slightly across numerical platforms.

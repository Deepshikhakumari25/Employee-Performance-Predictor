# Images

Dashboard and repository captures are saved in `screenshots/`. Screenshots should show actual application or GitHub output; do not fabricate missing views.

## Capture the test-results screenshot

`16_test_results.png` is not included because a genuine terminal capture was unavailable. To capture it manually in VS Code on Windows:

1. Open the project folder in VS Code and open **Terminal → New Terminal**.
2. Run `.\.venv\Scripts\python.exe -m pytest -q` from the project root. Confirm pytest completes successfully before capturing.
3. Keep the terminal output showing the final passing/skipped/failed summary visible. Widen the terminal if the summary wraps.
4. Press **Windows+Shift+S**, select the terminal area, and save the screenshot as `images\screenshots\16_test_results.png`.
5. Open the saved image and confirm it shows the actual command output clearly; replace it if the run failed or the result is unreadable.

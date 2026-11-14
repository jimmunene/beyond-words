# Current Status and Roadmap

## Source provenance

This repository is based on the team-provided updated `braintype_starter (1).zip`. Its 30 original source files match the earlier starter byte-for-byte; the updated ZIP adds `TEAM_TASKS.md`, `.gitignore`, and placeholder directories. The `beyond-words` name and repository administration files were added to prepare for GitHub collaboration.

## Implemented in starter source

- Synthetic EEG stream generator and recorded-data replay.
- CCA classifier and debounce/margin selection logic.
- Pygame speller with postcard and patient phrase modes.
- Local word prediction and optional Gemini integration code.
- EEG-based postcard generator.
- ElevenLabs and offline speech code.
- Consent-dependent upload queue for manual Base44 posting.
- CSV metrics utility.

These files are code paths, not a claim of tested live hardware performance. Current `config.yaml` contains placeholders for the gallery URL and voice ID.

## Before calling the project TrustLoop

- [ ] Explicit approval state before speech and publish, including patient phrases.
- [ ] Separate EEG-decoded characters from AI suggestions and from approved text.
- [ ] Verified event log showing committed and rejected decisions.
- [ ] True rest/abstain evaluation with participant calibration and held-out trials.
- [ ] Reproducible baseline-versus-TrustLoop comparison.
- [ ] Human-reviewed warnings/consent for any flickering stimuli.

## High-priority tasks

1. Verify installation and run keyboard UI.
2. Run synthetic EEG -> classifier -> UI end to end on team hardware.
3. Inspect licensed SSVEP data and establish channel/frequency mapping.
4. Validate live headset acquisition in an approved environment.
5. Implement IntentLock / approval gate and automated tests.
6. Run benchmark and record verified metrics.
7. Implement/test gallery and speech workflow.
8. Record reliable demo with honest input-source labels.

## Known gaps

- No actual g.tec acquisition executable or data is bundled.
- No code in the original starter provides a clinical proof of concept.
- Base44 provider API endpoint is not implemented.
- The original docs contain proposed event times; confirm your actual submission deadline.
- UI requires pygame/display, so CI intentionally tests core logic without a monitor.

## Changes merged from the updated upload

- Added the team-authored `TEAM_TASKS.md`, updating the repository URL and branch workflow for `beyond-words`.
- Reviewed the uploaded `.gitignore` and merged safeguards with the existing privacy-preserving rules.
- Preserved every application source file unchanged.
- Added `docs/DATASETS.md`, a shared dataset/source guide from prior team planning.
- No live EEG collection or end-to-end graphical validation has been claimed.

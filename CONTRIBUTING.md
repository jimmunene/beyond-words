# Contributing to Beyond Words

## Our workflow

1. Create a GitHub issue with a concrete acceptance criterion.
2. Branch from `main` as `feature/<short-description>` or `fix/<short-description>`.
3. Keep changes narrowly scoped, and update documentation if behavior changes.
4. Run `python -m unittest discover -s tests -v` and `python -m compileall -q .`.
5. Open a PR; have another team member review it before merging.

## Team contracts

- The EEG stream interface and channels must match `config.yaml` and actual headset measurements.
- Do not rewrite or silently invent the user's intended message.
- Never claim synthetic/replay/keyboard input is live EEG.
- Never commit EEG recordings, private names, logs, generated public uploads without consent, API keys, or credentials.
- Any calculated metric should state the source data, participant count, split, and method.

## Commit examples

`feat: add approval state to speller`

`test: verify abstain behavior during rest`

`docs: document Unicorn LSL setup`

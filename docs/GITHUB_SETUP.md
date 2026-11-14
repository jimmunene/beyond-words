# GitHub Setup — Beyond Words

**Target repository:** `jimmunene/beyond-words` (or your chosen team organization).

The connected GitHub integration can write to repositories that already exist, but does not expose repository creation. At the latest check, `jimmunene/beyond-words` did not exist. Create an **empty** GitHub repository once, then push this project.

## Step 1 — Create the repository

1. Open https://github.com/new and sign in as `jimmunene` (or your team organization).
2. Name: `beyond-words`.
3. Description: `EEG-powered assistive communication, consent-first messages, and brainwave postcards.`
4. Choose **Private** initially if the team wishes to keep work internal during the hackathon. Change visibility later only when ready and permitted by event rules.
5. **Do not** check “Add README,” “Add .gitignore,” or “Choose a license.” Those are already handled here or pending team agreement.
6. Create repository.

## Step 2 — Upload the full project via Git

From the unzipped `beyond-words/` folder on a team member's laptop:

```bash
git init -b main
git add .
git status --short
git commit -m "chore: initialize Beyond Words hackathon starter"
git remote add origin https://github.com/jimmunene/beyond-words.git
git push -u origin main
```

If `git init -b main` is unsupported, use `git init && git branch -M main` before committing. If the remote already exists, use `git remote set-url origin ...` rather than `git remote add`. Configure `git config --global user.name` and `user.email` locally if Git asks for them. GitHub may require browser-based authentication, SSH, or a credential manager; never paste credentials into project files or chat.

**Alternative:** After the empty repo is created, ask ChatGPT to upload this starter to `jimmunene/beyond-words` through the connected GitHub app. Use one push method; don't combine web upload and local push with conflicting commits.

## Step 3 — Add the three other teammates

GitHub repository → **Settings → Collaborators and teams** (or **Access → Collaborators**) → Add people. Give `Write` access to teammates; retain repository admin control. Ask each teammate to accept the invitation.

## Step 4 — Branching workflow

```bash
git clone https://github.com/jimmunene/beyond-words.git
cd beyond-words
git checkout -b feature/eeg-decoder
# edit, test, commit
git add .
git commit -m "feat: evaluate EEG decoder"
git push -u origin feature/eeg-decoder
```

Open a pull request to `main`; run automated checks and request one teammate review. Suggested branches:
- `feature/eeg-acquisition`
- `feature/trustloop-speller`
- `feature/postcards-voice`
- `feature/gallery-deployment`

Avoid sharing one branch among all four teammates.

## Step 5 — Configure GitHub

- **About:** use project description and topics `hackathon`, `eeg`, `bci`, `ssvep`, `accessibility`, `pygame`, `python`.
- **Actions:** ensure `.github/workflows/python-checks.yml` runs on pull requests.
- **Rules:** if available, require successful checks before merging into `main` (do not block urgent fixes without a maintainer override).
- **Issues:** create four milestones or issues for acquisition, TrustLoop/IntentLock, gallery/voice, and verification/demo.
- **Secrets:** `GEMINI_API_KEY` and `ELEVENLABS_API_KEY` should be placed only in GitHub Actions Secrets if a workflow actually needs them. The existing checks do not require them.
- **Demo:** disclose which runs are live EEG, synthetic, keyboard, or replay data.

## How to verify the push

Open https://github.com/jimmunene/beyond-words and check for `README.md`, `config.yaml`, `speller/`, `processing/`, `acquisition/`, `tests/`, and Actions. Do not upload the original starter ZIP as your only repository content: commit its extracted files.

## Updated starter merge

The uploaded `braintype_starter (1).zip` introduced `TEAM_TASKS.md`, a `.gitignore`, and two new placeholder directories. This package includes those changes and carries forward the GitHub-ready documentation and tests. See [Team Tasks](../TEAM_TASKS.md).

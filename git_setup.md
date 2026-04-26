# Git setup

This repo is not yet a git repository. Once you've extracted the zip, follow these steps to initialise it and push to a remote.

## 1. Initialise locally

```bash
cd shortcut-learning-lstm
git init
git add .
git commit -m "Initial scaffold"
```

## 2. Create a remote on GitHub

Go to https://github.com/new and create an empty repo named `shortcut-learning-lstm`. **Do not** initialise it with a README, .gitignore, or licence — those would conflict with the scaffold.

## 3. Connect and push

GitHub will show you the exact commands. They look like:

```bash
git remote add origin git@github.com:YOUR-USERNAME/shortcut-learning-lstm.git
git branch -M main
git push -u origin main
```

If you prefer HTTPS over SSH, use `https://github.com/YOUR-USERNAME/shortcut-learning-lstm.git` instead.

## 4. Add your team as collaborators

In the repo on GitHub: Settings → Collaborators → Add people. Add the other two team members by their GitHub usernames.

## 5. Set up branch protection (optional but recommended)

Settings → Branches → Add rule for `main`:
- Require pull request reviews before merging
- Require at least 1 approval

This enforces the workflow described in `CONTRIBUTING.md`.

## 6. Each team member clones

```bash
git clone git@github.com:YOUR-USERNAME/shortcut-learning-lstm.git
cd shortcut-learning-lstm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/  # smoke check
```

## 7. Working in branches

```bash
git checkout -b p1/data-loading      # Person 1's first branch
# ... do work ...
git add .
git commit -m "Implement load_imdb"
git push -u origin p1/data-loading
```

Then open a pull request on GitHub and request review.

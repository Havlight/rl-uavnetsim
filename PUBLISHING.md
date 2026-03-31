# Publishing `rl-uavnetsim`

## 1. Create the GitHub repository

Create an empty repository at:

`https://github.com/Havlight/rl-uavnetsim`

Do not initialize it with a README, `.gitignore`, or license, since those files already exist here.

## 2. Push this local directory

```bash
cd standalone_rl_uavnetsim
git init
git add .
git commit -m "Initial standalone rl_uavnetsim"
git branch -M main
git remote add origin https://github.com/Havlight/rl-uavnetsim.git
git push -u origin main
```

## 3. Alternative with GitHub CLI

```bash
cd standalone_rl_uavnetsim
gh repo create Havlight/rl-uavnetsim --public --source=. --remote=origin --push
```

## 4. Verify

After pushing, confirm:
- package sources appear under `rl_uavnetsim/`
- tests appear under `tests/`
- `README.md` renders correctly
- `implementation_plan3_5.md` is included for design reference

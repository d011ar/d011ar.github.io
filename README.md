# Meiyuan's Blog

This repository is a GitHub Pages + Jekyll blog for a personal site and posts.

## What is included

- `index.md`: homepage with profile and focus areas
- `about.md`: a fuller self-introduction page
- `posts.md`: a dedicated posts index page
- `_posts/`: all posts
- `_config.yml`: GitHub Pages / Jekyll configuration

## Before publishing

1. The repository name must be `user_name.github.io`.
2. Update `_config.yml`:
   - `title`
   - `email` if needed
3. Replace the placeholder project descriptions in `index.md`.
4. Replace the profile text in `about.md`.

## Create the GitHub repository

1. Log in to GitHub.
2. Create a new public repository named `user_name.github.io`.
3. Upload the contents of this folder to that repository.
4. Open `Settings -> Pages`.
5. Under `Build and deployment`, choose:
   - `Source`: `Deploy from a branch`
   - `Branch`: `main`
   - `Folder`: `/(root)`

## Suggested first edits

- Add a profile photo or CV PDF under `assets/`
- Add your project links on the homepage
- Add your first real post under `_posts/`

## Local git commands

```powershell
git init -b main
git add .
git commit -m "Initial personal blog"
git remote add origin https://github.com/user_name/user_name.github.io.git
git push -u origin main
```

## Notes

- This repository is designed to work with GitHub Pages using the `minima` theme.
- GitHub Pages may take a few minutes to publish updates after each push.

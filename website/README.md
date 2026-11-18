# Project website

This directory contains the static landing page for `sharedagents.ai`. The
documentation is built from the same checkout and served under `/docs/`.
The page uses local assets, with no analytics, external fonts, or runtime secrets.

Install the `docs` extra in your development environment, then run from the
repository root:

```bash
python website/build.py
python scripts/check_website.py
python -m http.server 8080 --bind 127.0.0.1 --directory build/website
```

Open `http://127.0.0.1:8080` and verify the desktop and mobile layouts, screenshot
switcher, image dialog, installation-command copy button, and documentation links.
`build/website/` is generated output and is ignored by Git.
Each build starts in a fresh staging directory. A previous output is preserved
under `build/website-previous-…` so removed assets cannot survive into a new
deployment. An alternate `--out` must also be a subdirectory of `build/`.

The automated browser checks need Playwright and Chromium, installed using the
same commands as [documentation captures](../docs/development/setup.md#capture-interface-images).

The interface images come from [the documentation capture script](../scripts/capture_docs.py).
They show the shipped interface with labeled demo data. Regenerate and review
them before changing the public gallery; never add screenshots of a real user's
sessions or account usage.

Website source changes follow the same branch, pull request, and review process
as application changes. Deploy only a reviewed build, and keep hosting credentials
and account identifiers in the hosting environment. DNS and certificate validation
must be complete before announcing a custom domain as live.

## GitHub Pages publication

The [website workflow](../.github/workflows/website.yml) builds and checks the
website on pull requests. Only `main` can deploy, using the `github-pages`
environment and GitHub's temporary deployment identity. It uploads only
`build/website/`, not the checkout, test reports, or local configuration.

For the initial deployment, a repository administrator must enable GitHub Pages
with **GitHub Actions** as the publishing source, set the custom domain to
`sharedagents.ai`, and verify domain ownership. Configure the domain on GitHub
before changing DNS. Preserve existing mail and verification records when
updating the apex website records; use the current targets from
[GitHub's custom-domain guide](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site).
Enable HTTPS once certificate provisioning completes, and verify both the home
page and `/docs/` before announcing publication.

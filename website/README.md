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

## Cloudflare Pages Direct Upload

The existing static build can be uploaded directly without a Git provider
integration. Use the intended account and the Free plan; keep account identifiers
and credentials outside the repository. Direct Upload cannot later be converted
to native Git integration in the same Pages project; external CI uploads remain
possible. See [Cloudflare's Direct Upload guide](https://developers.cloudflare.com/pages/get-started/direct-upload/).

Before uploading, run:

```bash
python -m pytest -q tests/test_website_publication.py
python website/build.py
python scripts/check_website.py
```

The build includes the unchanged root `install.sh`, a root `404.html`, and
MkDocs documentation, search index, and automatic sitemap under `/docs/`.
The root error page prevents Pages from treating missing paths as SPA routes.
The checker compares installer bytes with source without executing the script,
validates the sitemap against `mkdocs.yml` and `robots.txt`, and checks local
missing routes. Its local HTTP server does not emulate Cloudflare routing.

After independent review, create a Direct Upload project in **Workers & Pages**
and upload only the contents of `build/website/`, with `index.html` at the root.
Never upload the checkout, `.env` files, build environment, or QA reports.
Count files and check sizes first: dashboard drag-and-drop supports 1,000 files,
with a maximum size of 25 MiB per file. For a larger build, use another supported
Direct Upload method after review. Keep the source commit, artifact SHA-256,
per-file hashes, and resulting deployment identifier for review and rollback.

Test the `pages.dev` deployment before changing DNS. Verify actual home-page
content, mobile layout, gallery controls, documentation and search, stylesheet
and image loads, `/robots.txt`, `/docs/sitemap.xml`, and exact `/install.sh`
bytes with a non-HTML content type. Unknown paths at both root and under `/docs/`
must return 404, not the home page with status 200. Retest these checks on the
custom domain after its separately reviewed activation.

### Account and DNS gates

Before zone work, the owner must confirm password custody in a password manager,
working verification/recovery email, MFA enrollment, and securely stored recovery
codes. The owner performs email and registrar verification privately; automation
must not inspect mailbox content, password values, or recovery codes.

Before nameserver migration, export the complete registrar zone and retain its
nameservers, forwarding settings, TTLs, DNSSEC/DS state, and delegated subdomains
privately. Public DNS queries and Cloudflare's quick scan are incomplete inventories.
Reconcile every record, preserving mail/verification records and priorities, with
mail-related hostnames DNS-only. If restrictive CAA appears in the full export,
review the issuer requirements and exact proposed change before TLS provisioning.
Preserve the old zone for rollback. Keep the registrar and mail provider in place.

An apex Pages domain requires authoritative Cloudflare DNS in the same account
as the project. Follow [Cloudflare's DNSSEC migration sequence](https://developers.cloudflare.com/dns/zone-setups/full-setup/setup/):
an old provider's DS must not survive a delegation change to incompatible keys.
Enable new DNSSEC only after the new zone is stable; a delegation rollback must
also account for any newly published DS and resolver TTLs.

The canonical origin is `https://sharedagents.ai`, aligned with MkDocs `site_url`.
During the separately reviewed domain activation, associate the required Pages
hostnames and configure a Cloudflare zone redirect matching **only**
`www.sharedagents.ai` to the HTTPS apex, retaining the path and query string.
Do not use a catch-all redirect that captures the apex or `pages.dev` preview.
Pages `_redirects` does not support domain-level redirects; use Cloudflare's
[www-to-apex configuration](https://developers.cloudflare.com/pages/how-to/www-redirect/).
Verify a nested path with a query string, both hostnames' TLS certificates,
HTTP-to-HTTPS behavior, and unchanged mail DNS before announcing publication.

For a website regression, restore the prior reviewed Pages deployment. For a
DNS/mail regression, restore the exact saved records or old delegation with
the appropriate DNSSEC sequence; propagation is not instantaneous. An initial
publication can restore the saved website parking records if needed.

## GitHub Pages publication (alternative)

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

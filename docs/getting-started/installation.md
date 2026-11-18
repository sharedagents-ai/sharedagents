# Installation

## Requirements

Use Python 3.10–3.14 and a model provider or an OpenAI-compatible local model
server. Optional integrations have additional requirements: WhatsApp needs
Python 3.11+ and libmagic; local speech models and model servers may need
additional disk space or hardware.

## Install from PyPI

```bash
python -m pip install "sharedagents[web]"
sharedagents
```

The first run walks through endpoint, model, and agent setup. Provider secrets
belong in the runtime secret store or process environment. Do not put them in
a source checkout. See [configuration](configuration.md).

For a terminal-only installation, use `python -m pip install sharedagents`.
Add extras later with the same interpreter, for example
`python -m pip install "sharedagents[stt]"`.

## Browser and foreground stack

Once the runtime is configured and no existing SharedAgents components occupy
the default ports, run:

```bash
python -m sharedagents.serve
```

The launcher runs the orchestrator, gateway, and browser UI in the foreground.
It refuses occupied ports instead of adopting an unrelated process. Create a
browser login in another terminal:

```bash
sharedagents-web --create-user user@example.com
```

The command prompts for the password. Open `http://127.0.0.1:8110` and sign in.
The browser server requires a random web secret; the normal launcher manages
the stack's startup configuration. If starting the web server directly, follow
the [deployment guide](../guides/deployment.md).

## From source

```bash
git clone https://github.com/sharedagents-ai/sharedagents.git
cd sharedagents
python -m venv .venv
```

Activate the environment (`. .venv/bin/activate` on POSIX, or
`.venv\Scripts\Activate.ps1` in PowerShell), then run:

```bash
python -m pip install -e ".[web]"
sharedagents
```

For contributing, use the [development setup](../development/setup.md).
The root `sharedagents` launcher can also bootstrap its own virtual environment;
explicit environment setup makes the interpreter and installed extras visible.

## Upgrade

Back up the runtime home before upgrading, especially when release notes
describe a configuration or storage migration. Review the
[changelog](../../CHANGELOG.md), then upgrade through the same installation
method. Keep provider credentials and runtime backups outside the repository.

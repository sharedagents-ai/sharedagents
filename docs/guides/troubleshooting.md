# Troubleshooting

## Confirm the installation

Use the same Python environment that runs the application:

```bash
python --version
python -c "from importlib.metadata import version; print(version('sharedagents'))"
sharedagents --help
python -m sharedagents.serve --help
```

A shell command resolving to another environment can make an upgrade appear
ineffective. For source checkouts, confirm the intended branch and editable
install location before comparing behavior.

## A component port is occupied

The stack launcher does not adopt an existing process. Use
`python -m sharedagents.serve status` and identify which installation owns the
port. Use that installation's lifecycle command or service manager. Do not
start a second copy with a different identity or kill an unidentified process.

## Missing optional dependency

Install the matching extra into the application's environment, such as
`sharedagents[web]`, `sharedagents[stt]`, or `sharedagents[bedrock]`. WhatsApp
also needs libmagic. Wake-word classifiers are a separate explicit download;
see [voice models](voice-models.md).

## WebUI will not start or login is rejected

Direct web startup requires a random `SHAREDAGENTS_WEB_SECRET`. Browser
accounts are created through `sharedagents-web --create-user`, not by editing
a database. Repeated failed logins receive `429` and `Retry-After`; wait for
the indicated interval and check the proxy settings in [SECURITY.md](../../SECURITY.md).

## A model cannot be found or authenticated

Check that the configured model ID is actually available from its endpoint.
Keep endpoint aliases, served IDs, and provider credentials distinct. A model
row's `apiKeyEnv` names an environment variable; it is not the key value.
See [configuration](../getting-started/configuration.md) and the provider guide.

## Memory search seems stale

Use the incremental status/index commands in [memory maintenance](memory.md).
Markdown is the source of truth. A forced re-embedding pass can incur provider
costs and is a separate recovery operation, not the first diagnostic step.

## Ask for help

Follow [SUPPORT.md](../../SUPPORT.md). Include the version, installation method,
Python/OS, relevant optional extras, reproduction steps, and redacted error
messages. Do not upload the runtime home, raw conversation logs, databases,
QR codes, or environment files.

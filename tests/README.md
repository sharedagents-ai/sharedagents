# Tests

The default suite uses isolated runtime homes and sanitized fixtures. Test
files currently use a flat, behavior-oriented layout; find the relevant module
by subsystem (`test_web_*`, `test_gateway_*`, `test_*memory*`, and so on).

See [the testing guide](../docs/development/testing.md) for the default command,
optional browser/live selections, and publication checks. Keep regression
tests focused on observable behavior. Avoid production model calls, real
credentials, personal identifiers, or copied session dumps in fixtures.

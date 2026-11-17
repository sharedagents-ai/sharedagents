"""A systemctl down-verb on a glob that matches the gateway unit is refused."""

from sharedagents_agent.tools.safety import kills_sharedagents_gateway


def test_systemctl_glob_matching_the_gateway_unit_is_refused():
    for command in (
        "systemctl --user stop shared*",
        "systemctl --user stop s*",
        "systemctl --user mask sharedag?nts-gateway",
        "systemctl --user stop 'shared'*",
        "systemctl --user stop *",
    ):
        assert kills_sharedagents_gateway(command), command
    for command in (
        "systemctl --user stop foo*",
        "systemctl --user restart shared*",
        "systemctl --user stop sharedagents-web*",
    ):
        assert not kills_sharedagents_gateway(command), command

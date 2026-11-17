"""sharedagents_web.routes — Flask route blueprints (Task 2 de-monolith split).

Each submodule owns one cohesive route group extracted verbatim from
``server.py``; ``server.create_app()`` imports this package and registers every
blueprint in :data:`ALL_BLUEPRINTS` onto the module-level ``app`` singleton.

Import ordering note: this package is imported LAST — from inside
``server.register_blueprints()`` which runs at the very bottom of ``server``'s
module body — so each submodule's ``from sharedagents_web.server import <names>``
resolves against a fully-initialised ``server`` module (no import cycle).

authenticated users via the routes registered here.
"""

from sharedagents_web.routes.agent_builder import bp as agent_builder_bp
from sharedagents_web.routes.agents import bp as agents_bp
from sharedagents_web.routes.artifacts import bp as artifacts_bp
from sharedagents_web.routes.auth import bp as auth_bp
from sharedagents_web.routes.channels import bp as channels_bp
from sharedagents_web.routes.chat import bp as chat_bp
from sharedagents_web.routes.commands import bp as commands_bp
from sharedagents_web.routes.decision import bp as decision_bp
from sharedagents_web.routes.dev import bp as dev_bp
from sharedagents_web.routes.goals import bp as goals_bp
from sharedagents_web.routes.local import bp as local_bp
from sharedagents_web.routes.loops import bp as loops_bp
from sharedagents_web.routes.models import bp as models_bp
from sharedagents_web.routes.presence import bp as presence_bp
from sharedagents_web.routes.secrets import bp as secrets_bp
from sharedagents_web.routes.sessions import bp as sessions_bp
from sharedagents_web.routes.settings import bp as settings_bp
from sharedagents_web.routes.sharing import bp as sharing_bp
from sharedagents_web.routes.tasks import bp as tasks_bp
from sharedagents_web.routes.topics import bp as topics_bp
from sharedagents_web.routes.usage import bp as usage_bp
from sharedagents_web.routes.users import bp as users_bp
from sharedagents_web.routes.voice import bp as voice_bp

# Registration order is irrelevant to behavior (routes carry no url_prefix and
# every path is unique), but kept stable for readable ``app.url_map`` dumps.
ALL_BLUEPRINTS = [
    auth_bp,
    artifacts_bp,
    settings_bp,
    presence_bp,
    sessions_bp,
    topics_bp,
    voice_bp,
    chat_bp,
    commands_bp,
    agents_bp,
    goals_bp,
    loops_bp,
    dev_bp,
    tasks_bp,
    agent_builder_bp,
    usage_bp,
    local_bp,
    models_bp,
    decision_bp,
    channels_bp,
    users_bp,
    sharing_bp,
    secrets_bp,
]

__all__ = ["ALL_BLUEPRINTS"]

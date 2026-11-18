"""Native research ontology and optional typed knowledge-MCP workflows."""

__all__ = ["KnowledgeClient", "ResearchWorkflow", "ResearchError"]


def __getattr__(name):
    # Storage workers need only stdlib application policies and the SharedData
    # SDK, not the native runtime's optional MCP/agent dependencies.
    if name in __all__:
        from . import workflow

        return getattr(workflow, name)
    raise AttributeError(name)

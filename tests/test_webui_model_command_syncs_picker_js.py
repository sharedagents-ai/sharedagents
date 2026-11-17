"""A typed ``/model <alias>`` must move the composer's model selection.

Regression: in a fresh chat ``/model dpsk`` switched the session, but the
composer kept the picker's default (local qwen). The first Send re-applied
qwen with DeepSeek's remembered effort and was refused ("Could not apply
model qwen…; invalid effort 'high'").
"""

from test_webui_goal_reset_new_chat_js import _js, needs_node


@needs_node
def test_model_state_patch_moves_effective_selection() -> None:
    out = _js(
        "Object.assign(state,{currentSessionId:'S1',agents:['jarvis'],currentAgent:'jarvis',"
        "pickerAgent:'jarvis',pickerModels:{jarvis:'qwen-x'},"
        "modelOptionsByAgent:{jarvis:[{id:'qwen-x'},{id:'dpsk',wireId:'deepseek-v4-flash'}]},"
        "modelSheet:{phase:'CLOSED',effective:{agent:'jarvis',alias:'qwen-x'},"
        "draft:{agent:'jarvis',alias:'qwen-x'}}});"
        "applyStatePatch({model:'dpsk'});"
        "({eff: state.modelSheet.effective.alias, picker: state.pickerModels.jarvis, cur: state.currentModel})",
        extra_files=("commands.js",),
    )
    assert out == {"eff": "dpsk", "picker": "dpsk", "cur": "dpsk"}

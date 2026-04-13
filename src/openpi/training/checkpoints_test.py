from openpi.models import vlm_backbone as _vlm_backbone
from openpi.training import checkpoints as _checkpoints


def test_remap_restored_params_item_maps_legacy_vlm_root():
    restored_item = {
        "params": {
            _vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT: {"llm": {"embedder": {"value": 1}}},
            "action_in_proj": {"kernel": {"value": 2}},
        }
    }
    reference_params = {
        _vlm_backbone.RUNTIME_VLM_ROOT: {"llm": {"embedder": {"value": 0}}},
        "action_in_proj": {"kernel": {"value": 0}},
    }

    remapped = _checkpoints._remap_restored_params_item(restored_item, reference_params)

    assert _vlm_backbone.RUNTIME_VLM_ROOT in remapped["params"]
    assert _vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT not in remapped["params"]
    assert remapped["params"]["action_in_proj"] == restored_item["params"]["action_in_proj"]


def test_looks_like_legacy_vlm_root_mismatch_only_triggers_for_runtime_qwen_params():
    exc = ValueError(
        "User-provided restore item and on-disk value metadata tree structures do not match: "
        "{'params': {'vlm_with_expert': Diff(lhs={}, rhs=None), 'PaliGemma': Diff(lhs=None, rhs={})}}"
    )

    assert _checkpoints._looks_like_legacy_vlm_root_mismatch(
        exc, {_vlm_backbone.RUNTIME_VLM_ROOT: {"llm": {}}}
    )
    assert not _checkpoints._looks_like_legacy_vlm_root_mismatch(
        exc, {_vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT: {"llm": {}}}
    )


def test_looks_like_opt_state_structure_mismatch():
    exc = ValueError(
        "User-provided restore item and on-disk value metadata tree structures do not match: "
        "{'opt_state': [None, [Diff(lhs=<class \\'list\\'>, rhs=<class \\'dict\\'>)]]}"
    )
    assert _checkpoints._looks_like_opt_state_structure_mismatch(exc)


def test_swap_vlm_root_in_tree_handles_nested_optimizer_slots():
    tree = {
        "opt_state": [
            None,
            [
                {
                    "mu": {_vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT: {"llm": {"embedder": {"value": 1}}}},
                    "nu": {_vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT: {"llm": {"embedder": {"value": 2}}}},
                }
            ],
        ]
    }

    remapped = _checkpoints._swap_vlm_root_in_tree(
        tree,
        source_root=_vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT,
        target_root=_vlm_backbone.RUNTIME_VLM_ROOT,
    )

    mu = remapped["opt_state"][1][0]["mu"]
    nu = remapped["opt_state"][1][0]["nu"]
    assert _vlm_backbone.RUNTIME_VLM_ROOT in mu
    assert _vlm_backbone.RUNTIME_VLM_ROOT in nu
    assert _vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT not in mu
    assert _vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT not in nu

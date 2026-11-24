import pytest
import graphics.ui_ninepatch_batch as mod

CASES = [
    ("slice_by_dpi", (1,)),
    ("stable_atlas", (1,)),
    ("double_buffer_vb", (1,)),
    ("anchor_unified", (1,)),
    ("defer_layout", (1,)),
    ("end_frame_check", (1,)),
    ("layout", (1,)),
    ("batch", (1,)),
    ("animate", (1,)),
    ("present", (1,)),
    ("validate_ui", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None

import pytest
import render.wet_surface_rain as mod

CASES = [
    ("puddle_mask", (0.5, 0.2)),
    ("humidity_reflect", (0.3, 0.7)),
    ("mask_blend", (0.4, 0.6)),
    ("rain_intensity", (1.0, 2.0)),
    ("edge_gradient", (0.1, 0.2)),
    ("vehicle_interaction", (1, 2)),
    ("evaporate", (0.5, 1.0)),
    ("scene_reflection", (1, 2)),
    ("night_atten", (0.5, 0.2)),
    ("drop_landing", (1, 2)),
    ("ripple_normal", (2.0, 1.0)),
    ("wet_roughness", (0.2, 0.8)),
    ("puddle_depth", (0.05, 0.1)),
    ("spray_emit", (1, 2)),
    ("reflect_probe", (1, 2)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None

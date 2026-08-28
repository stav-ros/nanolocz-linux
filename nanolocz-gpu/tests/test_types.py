from nanolocz.core import Frame, Localizations, Meta, ParticleStack


def test_core_contracts_can_be_constructed_without_backend_dependencies() -> None:
    meta = Meta(pixel_size=(0.5, 0.5), height_unit="nm", channel="height")
    frame = Frame(data=[[1.0]], meta=meta)
    localizations = Localizations(xy=[(0.0, 0.0)], frame_index=[0])
    stack = ParticleStack(data=[[[[1.0]]]], centers_xy=[(0.0, 0.0)], frame_index=[0])

    assert frame.meta.channel == "height"
    assert localizations.frame_index == [0]
    assert stack.frame_index == [0]

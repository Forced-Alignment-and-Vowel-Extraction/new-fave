from new_fave.patterns.common_processing import resolve_speaker


def test_resolve_speaker_int():
    demo, speakers = resolve_speaker(0)
    assert demo is None
    assert speakers == [0]


def test_resolve_speaker_all():
    demo, speakers = resolve_speaker("all")
    assert demo is None
    assert speakers == "all"


def test_resolve_speaker_list():
    demo, speakers = resolve_speaker([0, 1])
    assert demo is None
    assert speakers == [0, 1]

from new_fave.patterns.common_processing import resolve_speaker


def test_resolve_speaker_list():
    demo, speakers = resolve_speaker([0, 1])
    assert demo is None
    assert speakers == [0, 1]

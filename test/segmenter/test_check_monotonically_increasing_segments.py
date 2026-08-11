


def test_check_monotonically_increasing_segments():
    import pandas as pd
    from pyroads.segmenter._util.check_segmentation import check_monotonically_increasing_segments
    
    df = pd.DataFrame(
        columns=[ 'road', 'carriageway', 'slk_from', 'slk_to'],
        data=[
            ["H001", "L", 0.001, 0.005],
            ["H001", "L", 0.005, 0.006],
            ["H001", "L", 0.007, 0.008],
        ]
    )

    assert check_monotonically_increasing_segments(
        df,
        categories=['road', 'carriageway'],
        measure=('slk_from', 'slk_to'),
    )

    df = pd.DataFrame(
        columns=[ 'road', 'carriageway', 'slk_from', 'slk_to'],
        data=[
            ["H001", "L", 0.0010, 0.0050],
            ["H001", "L", 0.0050, 0.0060],
            ["H001", "L", 0.0070, 0.0080],
            ["H002", "L", 0.0070, 0.0080],
            ["H003", "L", 0.0010, 0.0020],
            ["H003", "L", 0.0023, 0.0030],
            ["H003", "L", 0.0030, 0.0040],
        ]
    )
    assert check_monotonically_increasing_segments(
        df,
        categories=['road', 'carriageway'],
        measure=('slk_from', 'slk_to'),
    )

    df = pd.DataFrame(
        columns=[ 'road', 'carriageway', 'slk_from', 'slk_to'],
        data=[
            ["H001", "L", 0.0010, 0.0050],
            ["H001", "L", 0.0040, 0.0060],
            ["H001", "L", 0.0070, 0.0080],
            ["H002", "L", 0.0070, 0.0080],
            ["H003", "L", 0.0010, 0.0020],
            ["H003", "L", 0.0023, 0.0030],
            ["H003", "L", 0.0030, 0.0040],
        ]
    )
    assert not check_monotonically_increasing_segments(
        df,
        categories=['road', 'carriageway'],
        measure=('slk_from', 'slk_to'),
    )

def test_reversed_segments():
    import pandas as pd
    from pyroads.segmenter._util.check_segmentation import check_no_reversed_segments
    
    df = pd.DataFrame(
        columns=[ 'road', 'carriageway', 'slk_from', 'slk_to'],
        data=[
            ["H001", "L", 0.001, 0.005],
            ["H001", "L", 0.005, 0.006],
            ["H001", "L", 0.007, 0.008],
        ]
    )

    assert check_no_reversed_segments(
        df,
        measure=('slk_from', 'slk_to'),
    )

    df = pd.DataFrame(
        columns=[ 'road', 'carriageway', 'slk_from', 'slk_to'],
        data=[
            ["H001", "L", 0.005, 0.001],
            ["H001", "L", 0.005, 0.006],
            ["H001", "L", 0.007, 0.008],
        ]
    )

    assert not check_no_reversed_segments(
        df,
        measure=('slk_from', 'slk_to'),
    )
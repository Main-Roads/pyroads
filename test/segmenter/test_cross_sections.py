

def test_cross_sections():
    import pandas as pd
    from pyroads.segmenter import cross_sections
    data_to_iterate_over = pd.DataFrame(
        columns=["road_no", "carriageway", "xsp", "slk_from", "slk_to", "true_from", "true_to", "value"],
        data=[
            ["H001", "L", "L1", 0.010, 0.050, 0.010, 0.050, "a"], # SLK discontinuity (gap)
            ["H001", "L", "L1", 0.050, 0.070, 0.050, 0.070, "b"],
            ["H001", "L", "L1", 0.080, 0.100, 0.070, 0.090, "c"],
            ["H001", "L", "L1", 0.100, 0.120, 0.090, 0.110, "d"],

            ["H001", "L", "L2", 0.060, 0.070, 0.060, 0.070, "e"],
            ["H001", "L", "L3", 0.060, 0.070, 0.060, 0.070, "e"],

            ["H001", "R", "L1", 0.060, 0.070, 0.060, 0.070, "e"],
            ["H001", "R", "L2", 0.065, 0.070, 0.065, 0.070, "e"],

            ["H002", "L", "L1", 0.060, 0.075, 0.060, 0.075, "e"],
            ["H002", "L", "L2", 0.075, 0.080, 0.075, 0.080, "e"],
            ["H002", "R", "L1", 0.065, 0.078, 0.065, 0.078, "e"],
            ["H002", "R", "L2", 0.078, 0.085, 0.078, 0.085, "e"],
        ]
    )
    # note order of true and slk columns is swapped
    expected_result = pd.DataFrame(
        columns=["cross_section_number", "road_no", "carriageway", "xsp", "true_from",  "true_to",  "slk_from",  "slk_to",  "original_index",  "overlap"],
        data = [
            [0,    "H001",           "L",  "L1",      0.010,    0.060,     0.010,   0.060,               0,    0.040],
            [0,    "H001",           "L",  "L1",      0.010,    0.060,     0.010,   0.060,               1,    0.010],
            [1,    "H001",           "L",  "L1",      0.060,    0.065,     0.060,   0.065,               1,    0.005],
            [1,    "H001",           "L",  "L2",      0.060,    0.065,     0.060,   0.065,               4,    0.005],
            [1,    "H001",           "L",  "L3",      0.060,    0.065,     0.060,   0.065,               5,    0.005],
            [1,    "H001",           "R",  "L1",      0.060,    0.065,     0.060,   0.065,               6,    0.005],
            [2,    "H001",           "L",  "L1",      0.065,    0.070,     0.065,   0.070,               1,    0.005],
            [2,    "H001",           "L",  "L2",      0.065,    0.070,     0.065,   0.070,               4,    0.005],
            [2,    "H001",           "L",  "L3",      0.065,    0.070,     0.065,   0.070,               5,    0.005],
            [2,    "H001",           "R",  "L1",      0.065,    0.070,     0.065,   0.070,               6,    0.005],
            [2,    "H001",           "R",  "L2",      0.065,    0.070,     0.065,   0.070,               7,    0.005],
            [3,    "H001",           "L",  "L1",      0.070,    0.110,     0.080,   0.120,               2,    0.020],
            [3,    "H001",           "L",  "L1",      0.070,    0.110,     0.080,   0.120,               3,    0.020],

            [4,    "H002",           "L",  "L1",      0.060,    0.065,     0.060,   0.065,               8,    0.005],
            [5,    "H002",           "L",  "L1",      0.065,    0.075,     0.065,   0.075,               8,    0.010],
            [5,    "H002",           "R",  "L1",      0.065,    0.075,     0.065,   0.075,              10,    0.010],
            [6,    "H002",           "R",  "L1",      0.075,    0.078,     0.075,   0.078,              10,    0.003],
            [6,    "H002",           "L",  "L2",      0.075,    0.078,     0.075,   0.078,               9,    0.003],
            [7,    "H002",           "L",  "L2",      0.078,    0.080,     0.078,   0.080,               9,    0.002],
            [7,    "H002",           "R",  "L2",      0.078,    0.080,     0.078,   0.080,              11,    0.002],
            [8,    "H002",           "R",  "L2",      0.080,    0.085,     0.080,   0.085,              11,    0.005],
        ]
    )

    result = cross_sections(
        data_to_iterate_over,
        ["road_no"],
        ["carriageway", "xsp"],
        ("slk_from", "slk_to"),
        ("true_from", "true_to"),
    )

    pd.testing.assert_frame_equal(
        result,
        expected_result,
        check_like=True,
        check_exact=False,
        atol=0.0001, # absolute tolerance of 1 cm
    )

def test_cross_sections_preserves_index():
    import pandas as pd
    from pyroads.segmenter import cross_sections
    data_to_iterate_over = pd.DataFrame(
        columns=["road_no", "carriageway", "xsp", "slk_from", "slk_to", "true_from", "true_to", "value"],
        data=[
            ["H001", "L", "L1", 0.010, 0.050, 0.010, 0.050, "a"], # SLK discontinuity (gap)
            ["H001", "L", "L1", 0.050, 0.070, 0.050, 0.070, "b"],
            ["H001", "L", "L1", 0.080, 0.100, 0.070, 0.090, "c"],
            ["H001", "L", "L1", 0.100, 0.120, 0.090, 0.110, "d"],
            ["H001", "L", "L2", 0.060, 0.070, 0.060, 0.070, "e"],
            ["H001", "L", "L3", 0.060, 0.070, 0.060, 0.070, "f"],
        ],
        
    ).set_index("value")
    # note order of true and slk columns is swapped
    expected_result = pd.DataFrame(
        columns=["cross_section_number", "road_no", "carriageway", "xsp", "true_from",  "true_to",  "slk_from",  "slk_to",  "original_index",  "overlap"],
        data = [
            [0,    "H001",           "L",  "L1",       0.01,     0.06,      0.01,    0.06,              "a",     0.04],
            [0,    "H001",           "L",  "L1",       0.01,     0.06,      0.01,    0.06,              "b",     0.01],
            [1,    "H001",           "L",  "L1",       0.06,     0.07,      0.06,    0.07,              "b",     0.01],
            [1,    "H001",           "L",  "L2",       0.06,     0.07,      0.06,    0.07,              "e",     0.01],
            [1,    "H001",           "L",  "L3",       0.06,     0.07,      0.06,    0.07,              "f",     0.01],
            [2,    "H001",           "L",  "L1",       0.07,     0.11,      0.08,    0.12,              "c",     0.02],
            [2,    "H001",           "L",  "L1",       0.07,     0.11,      0.08,    0.12,              "d",     0.02],
        ]
    )

    result = cross_sections(
        data_to_iterate_over,
        ["road_no"],
        ["carriageway", "xsp"],
        ("slk_from", "slk_to"),
        ("true_from", "true_to"),
    )

    pd.testing.assert_frame_equal(
        result,
        expected_result,
        check_like=True,
        check_exact=False,
        atol=0.0001, # absolute tolerance of 1 cm
    )

if __name__ == "__main__":
    test_cross_sections()
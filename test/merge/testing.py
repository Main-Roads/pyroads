import pandas as pd
import pyroads.merge.merge as merge

# test group by

segments = pd.DataFrame(
    columns=["road", "cwy", "slk_from", "slk_to"],
    data=[
        ["H001", "L", 10, 20],
        ["H001", "L", 20, 30],
        ["H001", "L", 40, 50],
        ["H001", "L", 60, 70],
        ["H001", "R", 10, 20],
        ["H001", "R", 30, 50],
        ["H001", "R", 50, 70],
        ["H015", "L", 00, 10],
        ["H015", "L", 20, 40],
        ["H015", "L", 40, 60],
        ["H015", "R", 00, 30],
        ["H015", "R", 30, 40],
        ["H015", "R", 40, 60],
        ["H016", "L", 00, 60],
        ["H018", "L", 00, 60],
    ],
)


data = pd.DataFrame(
    columns=["road", "cwy", "slk_from", "slk_to", "measure_a", "cat_1"],
    data=[
        ["H001", "L", 00, 10, 0.0, "A"],
        ["H001", "L", 10, 20, 1.0, "B"],
        ["H001", "L", 20, 40, 2.0, "C"],
        ["H001", "L", 40, 70, 3.0, "D"],
        ["H001", "R", 10, 20, 5.0, "E"],
        ["H001", "R", 30, 40, 6.0, "F"],
        ["H001", "R", 40, 45, 6.0, "G"],
        ["H001", "R", 45, 55, 6.0, "H"],
        ["H001", "R", 55, 80, 7.0, "I"],
        ["H015", "L", 00, 4, 1.0, "J"],
        ["H015", "L", 4, 10, None, "K"],
        ["H015", "L", 10, 35, 8.0, "L"],
        ["H015", "L", 30, 50, 9.0, "M"],
        ["H015", "L", 40, 50, 10.0, "N"],
        ["H015", "L", 50, 80, 10.0, "O"],
        ["H015", "R", 00, 20, 11.0, "P"],
        ["H015", "R", 30, 40, None, "Q"],
        ["H015", "R", 40, 60, 13.0, "R"],
        ["H016", "L", 00, 30, 14.0, "S"],
    ],
)

res = merge.on_slk_intervals(
    segments,
    data,
    ["road", "cwy"],
    [
        merge.Action(
            "measure_a", rename="longest", aggregation=merge.Aggregation.KeepLongest()
        ),
        merge.Action(
            "measure_a", rename="mean", aggregation=merge.Aggregation.Average()
        ),
        merge.Action(
            "measure_a",
            rename="lenw_mean",
            aggregation=merge.Aggregation.LengthWeightedAverage(),
        ),
        merge.Action(
            "measure_a",
            rename="lenw_prc75",
            aggregation=merge.Aggregation.LengthWeightedPercentile(0.75),
        ),
        merge.Action(
            "cat_1", rename="cat", aggregation=merge.Aggregation.KeepLongest()
        ),
    ],
    from_to=("slk_from", "slk_to"),
)

print(res)
print(res.dtypes)
exit()
segments["segment_id"] = segments.index
segments = segments.set_index(["road", "cwy", "segment_id"])
segments.sort_index()
# print(segments)

data["data_id"] = data.index
data = data.set_index(["road", "cwy", "data_id"])
data = data.sort_index()
# print(data)

for segment_group_index, segment_group in segments.groupby(level=[0, 1]):
    print("\n\n==========================")
    print(f"segment_group_index {segment_group_index}")
    print(f"segment_group {segment_group}")
    print(f"type(segment_group) {type(segment_group)}")
    print(f"Dataframe: {isinstance(segment_group, pd.DataFrame)}")
    print(f"Series: {isinstance(segment_group, pd.Series)}")

    try:
        data_to_merge = data.loc[segment_group_index]
    except KeyError:
        # no data under that key. Skip
        continue
    print(data_to_merge)

    for segment_row_index, segment_row in segment_group.iterrows():
        print("\n==========")
        print(f"segment_row_index, {segment_row_index}")
        print(f"segment_row, {segment_row}")
        print(f"type(segment_row) {type(segment_row)}")

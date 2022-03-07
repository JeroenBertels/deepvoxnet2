import os
from deepvoxnet2 import DEMO_DIR
from deepvoxnet2.components.mirc import Mirc, Dataset, Case, Record, NiftiFileModality, NiftiFileMultiModality


def create_dataset(data="train"):
    if data == "train":
        case_names = ["case_{}".format(i) for i in range(6)]

    elif data == "val":
        case_names = ["case_{}".format(i) for i in range(6, 8)]

    elif data == "test":
        case_names = ["case_{}".format(i) for i in range(8, 10)]

    else:
        raise ValueError("data must be 'train', 'val' or 'test'")

    dataset = Dataset("brats_2018_" + data, os.path.join(DEMO_DIR, "brats_2018"))
    for case_name in case_names:
        case = Case(case_name)
        record = Record("record_0")
        record.add(NiftiFileModality("flair", os.path.join(DEMO_DIR, "brats_2018", case_name, "FLAIR.nii.gz")))
        record.add(NiftiFileModality("t1", os.path.join(DEMO_DIR, "brats_2018", case_name, "T1.nii.gz")))
        record.add(NiftiFileMultiModality("flair-t1", [os.path.join(DEMO_DIR, "brats_2018", case_name, f"{mod}.nii.gz") for mod in ["FLAIR", "T1"]]))
        record.add(NiftiFileModality("whole_tumor", os.path.join(DEMO_DIR, "brats_2018", case_name, "GT_W.nii.gz")))
        case.add(record)
        dataset.add(case)

    return dataset


if __name__ == '__main__':
    brats_2018_train = create_dataset("train")
    brats_2018_val = create_dataset("val")
    brats_2018_test = create_dataset("test")

    # When you add datasets to a "higher level" Mirc object, you can have nice new baked in methods:
    mirc = Mirc()
    mirc.add(brats_2018_train)
    df = mirc.get_df("flair")
    print(df)
    u, s = mirc.mean_and_std("t1")
    print(u, s)
    mirc.inspect(["flair", "t1"])

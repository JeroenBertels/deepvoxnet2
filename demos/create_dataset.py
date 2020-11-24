import os
from deepvoxnet2 import DEMO_DIR
from deepvoxnet2.components.mirc import Dataset, Case, Record, NiftyFileModality


def create_dataset(data="train"):
    if data == "train":
        case_names = ["case_{}".format(i) for i in range(8)]

    elif data == "val":
        case_names = ["case_{}".format(i) for i in range(8, 10)]

    else:
        raise ValueError

    dataset = Dataset("brats_2018_" + data, os.path.join(DEMO_DIR, "brats_2018"))
    for case_name in case_names:
        case = Case(case_name)
        record = Record("record_0")
        record.add(NiftyFileModality("flair", os.path.join(DEMO_DIR, "brats_2018", case_name, "FLAIR.nii.gz")))
        record.add(NiftyFileModality("t1", os.path.join(DEMO_DIR, "brats_2018", case_name, "T1.nii.gz")))
        record.add(NiftyFileModality("whole_tumor", os.path.join(DEMO_DIR, "brats_2018", case_name, "GT_W.nii.gz")))
        case.add(record)
        dataset.add(case)

    return dataset


if __name__ == '__main__':
    brats_2018_train = create_dataset()
    brats_2018_val = create_dataset('val')

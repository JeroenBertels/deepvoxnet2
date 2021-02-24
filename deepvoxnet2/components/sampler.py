import random


class Identifier(dict):
    def __init__(self, **kwargs):
        super(Identifier, self).__init__()
        for key in kwargs:
            self[key] = kwargs[key]

    def __call__(self):
        raise NotImplementedError


class MircIdentifier(dict):
    def __init__(self, mirc=None, dataset_id=None, case_id=None, record_id=None, modality_id=None):
        super(MircIdentifier, self).__init__(mirc=mirc, dataset_id=dataset_id, case_id=case_id, record_id=record_id, modality_id=modality_id)
        self.mirc = mirc
        self.dataset_id = dataset_id
        self.case_id = case_id
        self.record_id = record_id
        self.modality_id = modality_id

    def __call__(self):
        return self.dataset_id, self.case_id, self.record_id, self.modality_id


class SampleIdentifier(dict):
    def __init__(self, sample=None):
        super(SampleIdentifier, self).__init__(sample=sample)
        self.sample = sample

    def __call__(self):
        return "sample"


class Sampler(object):
    def __init__(self, identifiers, shuffle=False):
        self.identifiers = identifiers
        self.shuffle = shuffle
        self.randomize()

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, idx):
        return self.identifiers[idx]

    def __iter__(self):
        return iter(self.identifiers)

    def __add__(self, other):
        assert self.shuffle == other.shuffle
        return Sampler(self.identifiers + other.identifiers, shuffle=self.shuffle)

    def randomize(self):
        if self.shuffle:
            random.shuffle(self.identifiers)


class MircSampler(Sampler):
    def __init__(self, mirc, mode="per_record", **kwargs):
        if mode == "per_record":
            identifiers = [MircIdentifier(mirc, dataset_id, case_id, record_id) for dataset_id in mirc for case_id in mirc[dataset_id] for record_id in mirc[dataset_id][case_id]]

        else:
            raise NotImplementedError

        super(MircSampler, self).__init__(identifiers, **kwargs)


class SampleSampler(Sampler):
    def __init__(self, samples, **kwargs):
        identifiers = [SampleIdentifier(sample) for sample in samples]
        super(SampleSampler, self).__init__(identifiers, **kwargs)

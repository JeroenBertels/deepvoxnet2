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
    def __init__(self, identifiers=None, shuffle=False, weights=None):
        self.identifiers = [] if identifiers is None else identifiers
        self.base_identifiers = self.identifiers
        self.shuffle = shuffle
        self.weights = weights
        self.randomize()

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, idx):
        return self.identifiers[idx]

    def __iter__(self):
        return iter(self.identifiers)

    def randomize(self):
        self._randomize()
        if self.shuffle:
            if self.weights is None:
                random.shuffle(self.base_identifiers)

            else:
                _ = list(zip(self.base_identifiers, self.weights))
                random.shuffle(_)
                self.base_identifiers, self.weights = zip(*_)

        if len(self.base_identifiers) > 0 and self.weights is not None:
            assert len(self.base_identifiers) == len(self.weights), "When sample weights are specified, you must specify a weight for each identifier."
            self.identifiers = random.choices(self.base_identifiers, weights=self.weights, k=len(self.base_identifiers))

        else:
            self.identifiers = self.base_identifiers

    def _randomize(self):
        pass


class MircSampler(Sampler):
    def __init__(self, mirc, mode="per_record", **kwargs):
        self.mirc = mirc
        self.mode = mode
        super(MircSampler, self).__init__(**kwargs)

    def _randomize(self):
        self.base_identifiers = []
        if self.mode == "per_record":
            for dataset_id in self.mirc:
                for case_id in self.mirc[dataset_id]:
                    for record_id in self.mirc[dataset_id][case_id]:
                        self.base_identifiers.append(MircIdentifier(self.mirc, dataset_id, case_id, record_id))

        elif self.mode == "per_case":
            for dataset_id in self.mirc:
                for case_id in self.mirc[dataset_id]:
                    record_id_i = random.randint(0, len(self.mirc[dataset_id][case_id]) - 1)
                    for i, record_id in enumerate(self.mirc[dataset_id][case_id]):
                        if i == record_id_i:
                            self.base_identifiers.append(MircIdentifier(self.mirc, dataset_id, case_id, record_id))
                            break

        else:
            raise NotImplementedError


class SampleSampler(Sampler):
    def __init__(self, samples, **kwargs):
        identifiers = [SampleIdentifier(sample) for sample in samples]
        super(SampleSampler, self).__init__(identifiers, **kwargs)

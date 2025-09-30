import logging
from collections import Counter

class BalancerQuality:
    def __init__(self, scale=1.0):
        """
        Parameters:
        - scale: multiplier for dataset size (e.g. 2.0 doubles, 0.5 halves)
        """
        self.scale = scale

    def balance_labels(self, ds):
        """
        Decide how many observations to add per label to reach scale * dataset_size.
        Does NOT filter by similarity (that’s handled separately).
        """
        # Count positives per label
        d = {}
        for l in range(len(ds[0]["labels"])):
            d[l] = {"positive": 0, "negative": 0}

        for ex in ds:
            labels = [i for i, v in enumerate(ex["labels"]) if v == 1]
            for l in labels:
                d[l]["positive"] += 1

        # Compute scaling
        orig_size = len(ds)
        target_size = int(orig_size * self.scale)
        extra_needed = max(0, target_size - orig_size)

        labels = list(d.keys())
        added_pos = {l: 0 for l in labels}

        if extra_needed > 0:
            per_label = extra_needed // len(labels)
            for l in labels:
                added_pos[l] = per_label
            remainder = extra_needed % len(labels)
            for l in labels[:remainder]:
                added_pos[l] += 1

        # Final structure
        result = {}
        resultL = []
        for l in labels:
            final_pos = d[l]["positive"] + added_pos[l]
            result[l] = {
                "add": added_pos[l],
                "positive": final_pos,
                "negative": d[l]["negative"],
            }
            resultL.append(added_pos[l])

        logging.info(result)
        return resultL

    def __call__(self, ds, lang=None):
        return self.balance_labels(ds)

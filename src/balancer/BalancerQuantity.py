import logging
from collections import Counter


class BalancerQuantity:
    def __init__(self, scale=1.0):
        """
        Parameters:
        - scale: multiplier for dataset size (e.g. 2.0 doubles, 0.5 halves)
        """
        self.scale = scale
        self.max_iter = 10000
        self.langs = ["java", "python", "pharo"]
        self.labels = {
            "java": [
                "summary",
                "Ownership",
                "Expand",
                "usage",
                "Pointer",
                "deprecation",
                "rational",
            ],
            "python": ["Usage", "Parameters", "DevelopmentNotes", "Expand", "Summary"],
            "pharo": [
                "Keyimplementationpoints",
                "Example",
                "Responsibilities",
                "Intent",
                "Keymessages",
                "Collaborators",
            ],
        }

    def _split_list_into_columns(self, row):
        values_list = row["labels"]  # e.g., [0,0,0,1,0]
        return {str(i): v for i, v in enumerate(values_list)}

    def _get_data(self, ds):
        return ds.map(lambda row: self._split_list_into_columns(row))

    def balance_labels(self, ds):
        """
        Instead of balancing by ratio, just compute how many
        new samples are needed to reach scale * dataset_size.
        """
        data = self._get_data(ds)

        # Count positives/negatives per label
        d = {}
        for l in range(20):  # assuming up to 10 labels
            try:
                group = Counter(data[l])
                d[l] = {"positive": group[1], "negative": group[0]}
            except Exception:
                continue

        labels = list(d.keys())
        added_pos = {l: 0 for l in labels}

        # --- scaling ---
        orig_size = sum(v["positive"] + v["negative"] for v in d.values())
        target_size = int(orig_size * self.scale)
        extra_needed = target_size - orig_size

        # distribute extra_needed across labels
        if extra_needed > 0:
            # just spread evenly across labels
            per_label = extra_needed // len(labels)
            for l in labels:
                added_pos[l] = per_label
            # assign remainder randomly
            remainder = extra_needed % len(labels)
            for l in labels[:remainder]:
                added_pos[l] += 1
        elif extra_needed < 0:
            # removing samples: treat as negatives being reduced
            per_label = abs(extra_needed) // len(labels)
            for l in labels:
                d[l]["negative"] = max(0, d[l]["negative"] - per_label)
            remainder = abs(extra_needed) % len(labels)
            for l in labels[:remainder]:
                d[l]["negative"] = max(0, d[l]["negative"] - 1)

        # --- final counts ---
        result = {}
        resultL = []
        for l in labels:
            final_pos = d[l]["positive"] + added_pos[l]
            final_neg = d[l]["negative"] + sum(
                added_pos[other] for other in labels if other != l
            )
            result[l] = {
                "add": added_pos[l],
                "positive": final_pos,
                "negative": final_neg,
            }
            resultL.append(added_pos[l])

        logging.info(result)
        return resultL

    def __call__(self, ds, lang=None):
        return self.balance_labels(ds)

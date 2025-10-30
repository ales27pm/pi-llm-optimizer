"""Anonymization helpers for the Québecois French corpus blueprint."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

LOGGER = logging.getLogger("qf.anonymize")


def enforce_k_anonymity(
    records: Iterable[Dict[str, object]],
    *,
    quasi_identifiers: Sequence[str] = ("region_qc", "age_group"),
    k: int = 5,
) -> List[Dict[str, object]]:
    anonymized: List[Dict[str, object]] = []
    buckets: Dict[Tuple, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        sociolinguistic = dict(record.get("sociolinguistic_parameters", {}))
        key = tuple(sociolinguistic.get(field, "unspecified") for field in quasi_identifiers)
        buckets[key].append((dict(record), sociolinguistic))
    for key, bucket in buckets.items():
        if len(bucket) < k:
            LOGGER.debug("Bucket %s below k=%s; applying coarsening", key, k)
            for record, sociolinguistic in bucket:
                for field in quasi_identifiers:
                    sociolinguistic[field] = "unspecified"
                record["sociolinguistic_parameters"] = sociolinguistic
                anonymized.append(record)
        else:
            anonymized.extend(record for record, _ in bucket)
    return anonymized


def ensure_l_diversity(
    records: Iterable[Dict[str, object]],
    *,
    quasi_identifiers: Sequence[str] = ("region_qc", "age_group"),
    sensitive_field: str = "gender",
    l_threshold: int = 2,
) -> List[Dict[str, object]]:
    buckets: Dict[Tuple, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        sociolinguistic = dict(record.get("sociolinguistic_parameters", {}))
        key = tuple(sociolinguistic.get(field, "unspecified") for field in quasi_identifiers)
        buckets[key].append((record, sociolinguistic))

    diversified: List[Dict[str, object]] = []
    for key, bucket in buckets.items():
        values = {socio.get(sensitive_field, "unspecified") for _, socio in bucket}
        if len(values) < l_threshold:
            LOGGER.debug("Bucket %s below l=%s; masking %s", key, l_threshold, sensitive_field)
            for record, sociolinguistic in bucket:
                sociolinguistic[sensitive_field] = "unspecified"
                record["sociolinguistic_parameters"] = sociolinguistic
                diversified.append(record)
        else:
            diversified.extend(record for record, _ in bucket)
    return diversified


def audit_privacy_levels(
    records: Iterable[Dict[str, object]],
    *,
    quasi_identifiers: Sequence[str] = ("region_qc", "age_group"),
    sensitive_field: str = "gender",
) -> Dict[str, object]:
    buckets: Dict[Tuple, Counter] = defaultdict(Counter)
    for record in records:
        sociolinguistic = dict(record.get("sociolinguistic_parameters", {}))
        key = tuple(sociolinguistic.get(field, "unspecified") for field in quasi_identifiers)
        buckets[key][sociolinguistic.get(sensitive_field, "unspecified")] += 1
    summary = {
        "buckets": {
            ",".join(map(str, key)): dict(counter)
            for key, counter in sorted(buckets.items(), key=lambda item: (-sum(item[1].values()), item[0]))
        }
    }
    summary["min_bucket_size"] = min((sum(counter.values()) for counter in buckets.values()), default=0)
    summary["max_bucket_size"] = max((sum(counter.values()) for counter in buckets.values()), default=0)
    return summary


__all__ = ["enforce_k_anonymity", "ensure_l_diversity", "audit_privacy_levels"]

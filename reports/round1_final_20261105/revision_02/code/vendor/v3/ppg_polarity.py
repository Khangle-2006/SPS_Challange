"""Versioned PPG-only adapter boundary. Never changes images, clocks or HR labels."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np

POLICY_PATH = Path(__file__).with_name('POLARITY_POLICY.json')
POLICY = json.loads(POLICY_PATH.read_text(encoding='utf-8'))
POLICY_SHA256 = hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()
CONVENTION = 'blood_volume_positive_v1'
RAW = 'raw_device'
KEY = 'ppg_polarity'
ALIASES = {'MCD-rPPG': 'MCD', 'mcd': 'MCD', 'ubfc2': 'UBFC2',
           'ubfc1': 'UBFC1', 'pure': 'PURE', 'aipex': 'AiPEX', 'pilot': 'Pilot'}
SIGNS = {'MCD': -1, 'UBFC2': 1}


def dataset_name(dataset):
    return ALIASES.get(dataset, dataset)


def normalize_channels(channels, dataset, *, source_convention, provenance=None):
    """Return new PPG arrays and an exactly-once provenance receipt.

    Explicit source convention is mandatory for caches; unknown datasets refuse
    waveform use until a separately versioned adapter convention is approved.
    Existing canonical arrays require their receipt; they are never flipped again.
    """
    dataset = dataset_name(dataset)
    if dataset not in SIGNS:
        raise ValueError('Unverified PPG polarity convention: ' + str(dataset))
    if not channels or set(channels) - {'ppg', 'ppg_frame'}:
        raise ValueError('Expected ppg and/or ppg_frame channels only')
    if source_convention == CONVENTION:
        if not isinstance(provenance, dict) or any(provenance.get(k) != v for k, v in
                {'status': 'canonical', 'convention': CONVENTION, 'dataset': dataset,
                 'policy_sha256': POLICY_SHA256, 'raw_to_canonical_sign': SIGNS[dataset]}.items()):
            raise ValueError('Canonical PPG requires matching polarity provenance')
        if not set(channels) <= set(provenance.get('channels', [])):
            raise ValueError('Canonical provenance does not cover PPG channels')
        return {k: np.asarray(v).copy() for k, v in channels.items()}, dict(provenance)
    if source_convention != RAW:
        raise ValueError('Unknown source PPG convention')
    if provenance is not None:
        raise ValueError('Raw-device declaration with existing provenance; refuse double flip')
    result = {k: np.asarray(v, dtype=float) * SIGNS[dataset] for k, v in channels.items()}
    receipt = dict(status='canonical', convention=CONVENTION, dataset=dataset,
                   policy_sha256=POLICY_SHA256, raw_to_canonical_sign=SIGNS[dataset],
                   channels=sorted(channels), decision_cohort='development60_only',
                   application_scope='all_cameras_all_partitions_without_reselection')
    return result, receipt


def adapt_cached_meta(meta, dataset, *, source_convention, provenance=None):
    """Adapt an explicitly declared raw/canonical cache in RAM, before differencing.

    Caller persists the returned receipt in run/cache provenance. The source cache
    is never rewritten. Images, timestamps, validity masks and non-PPG fields retain
    the original objects/values. Do not apply a downstream target multiplier.
    """
    result = dict(meta)
    channels = {k: meta[k] for k in ('ppg', 'ppg_frame') if k in meta}
    normalized, receipt = normalize_channels(channels, dataset,
        source_convention=source_convention, provenance=provenance)
    result.update(normalized)
    return result, receipt


def normalize_recording(recording):
    """Duck-typed dataclass Recording boundary; repeated calls are idempotent."""
    dataset = dataset_name(recording.metadata.get('dataset'))
    channels = {k: v.values for k, v in recording.labels.items() if k in ('ppg', 'ppg_frame')}
    metadata = dict(recording.metadata)
    old = metadata.get(KEY)
    if not channels:
        metadata[KEY] = dict(status='not_applicable_no_ppg', dataset=dataset,
                             convention=None, policy_sha256=POLICY_SHA256)
        return replace(recording, metadata=metadata)
    if dataset not in SIGNS:
        metadata[KEY] = dict(status='unverified_requires_adapter_contract', dataset=dataset,
                             convention=None, policy_sha256=POLICY_SHA256)
        return replace(recording, metadata=metadata)
    values, receipt = normalize_channels(channels, dataset,
        source_convention=CONVENTION if old is not None else RAW, provenance=old)
    labels = dict(recording.labels)
    for name, value in values.items():
        labels[name] = replace(labels[name], values=value)
    metadata[KEY] = receipt
    return replace(recording, labels=labels, metadata=metadata)

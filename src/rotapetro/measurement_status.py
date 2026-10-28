"""V1 measurement-status conventions.

Every measured feature in V1 carries an explicit status alongside its value.
This replaces the legacy practice of using NaN/0/None to represent different
underlying states (observed zero, unresolved, not applicable, invalid).
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Any


class MeasurementStatus(str, Enum):
    """Status of a single measured feature."""
    OBSERVED = 'observed'
    NOT_DETECTED_WITH_BOUND = 'not_detected_with_bound'
    UNRESOLVED = 'unresolved'
    INVALID = 'invalid'
    NOT_APPLICABLE = 'not_applicable'


@dataclass
class MeasuredFeature:
    """A feature value with its status and quality metadata."""
    value: Any
    status: MeasurementStatus
    support_pixels: int = 0
    quality: float = 0.0  # e.g., R², or fraction of valid pixels
    bound: float | None = None  # upper bound for NOT_DETECTED
    note: str = ''

    @property
    def is_usable(self) -> bool:
        return self.status in (MeasurementStatus.OBSERVED,
                               MeasurementStatus.NOT_DETECTED_WITH_BOUND)

    def to_dict(self, prefix: str = '') -> dict:
        """Flatten to dict entries suitable for a DataFrame row."""
        p = f"{prefix}_" if prefix else ''
        d = {
            f"{p}value": self.value,
            f"{p}status": self.status.value,
            f"{p}support": self.support_pixels,
            f"{p}quality": self.quality,
        }
        if self.bound is not None:
            d[f"{p}bound"] = self.bound
        return d


def make_observed(value, support: int = 0, quality: float = 1.0) -> MeasuredFeature:
    return MeasuredFeature(value, MeasurementStatus.OBSERVED, support, quality)

def make_not_detected(bound: float, support: int = 0, quality: float = 1.0) -> MeasuredFeature:
    return MeasuredFeature(None, MeasurementStatus.NOT_DETECTED_WITH_BOUND, support, quality, bound)

def make_unresolved(support: int = 0, note: str = '') -> MeasuredFeature:
    return MeasuredFeature(None, MeasurementStatus.UNRESOLVED, support, note=note)

def make_invalid(note: str = '') -> MeasuredFeature:
    return MeasuredFeature(None, MeasurementStatus.INVALID, note=note)

def make_not_applicable(note: str = '') -> MeasuredFeature:
    return MeasuredFeature(None, MeasurementStatus.NOT_APPLICABLE, note=note)

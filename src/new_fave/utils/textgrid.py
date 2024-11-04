from aligned_textgrid import (AlignedTextGrid, 
    SequenceTier, 
    SequenceInterval
)
from aligned_textgrid.sequences.tiers import TierGroup
from fasttrackpy import CandidateTracks
import numpy as np

def get_top_tier(interval: SequenceInterval) -> SequenceTier:
    """Given a sequence interval, return the top-level tier
    in the hierarchy

    Args:
        interval (SequenceInterval):
            A sequence interval

    Returns:
        (SequenceTier):
            The top level sequence tier
    """
    if isinstance(interval.within, SequenceInterval):
        return get_top_tier(interval.within)
    return interval.within

def get_tier_group(interval: SequenceInterval)->TierGroup:
    """Given a SequenceInterval, returns the TierGroup
    it is within.

    Args:
        interval (SequenceInterval): A SequenceInterval

    Returns:
        (TierGroup): The containing TierGroup
    """
    top_tier = get_top_tier(interval)
    return top_tier.within

def get_textgrid(interval: SequenceInterval)->AlignedTextGrid:
    """Given a SequenceInterval, return the containing
    AlignedTextGrid object

    Args:
        interval (SequenceInterval): A SequenceInterval

    Returns:
        (AlignedTextGrid): The containing AlignedTextGrid
    """
    tier_group = get_tier_group(interval)
    return tier_group.within

def get_all_textgrid(
        candidates: list[CandidateTracks],
    ) -> list[AlignedTextGrid]:
    """Get all unique textgrids

    Args:
        candidates (list[CandidateTracks] | list[VowelMeasurement]):
            A list of either fasttrackpy `CandidateTracks` 
            or a list of new_fave `VowelMeasurement`s

    Returns:
        list[AlignedTextGrid]: _description_
    """
    return list({
        get_textgrid(cand.interval)
        for cand in candidates
    })

def mark_overlaps(
        atg: AlignedTextGrid
):
    """
    For all Phone intervals in a data frame, 
    mark whether or not they are overlapping with another tier's content.
    It adds a boolean `overlapped` attribute to the SequenceIntervals.

    Args:
        atg (AlignedTextGrid):
            The aligned textgrid on which to mark overaps.
    """
    for g in atg:
        for t in g:
            for i in t:
                i.set_feature("overlapped", False)

    if len(atg) == 1:
        return

    for i in range(len(atg)-1):
        for j in range(i+1, len(atg)):
            tier1 = atg[i].Phone
            tier2 = atg[j].Phone

            x1 = tier1.starts
            x2 = tier1.ends

            y1 = tier2.starts
            y2 = tier2.ends

            a = np.array([x >= y1 for x in x2])
            b = np.array([x <= y2 for x in x1])

            overlap_locs = np.where(a & b)

            for idx1, idx2 in zip(*overlap_locs):
                if tier1[idx1].label != "" and tier2[idx2].label != "":
                    tier1[idx1].overlapped = True
                    tier2[idx2].overlapped = True

            return overlap_locs
from aligned_textgrid import AlignedTextGrid, \
    SequenceTier, \
    SequenceInterval
from aligned_textgrid.sequences.tiers import TierGroup
from fasttrackpy import CandidateTracks
from new_fave.measurements.vowel_measurement import VowelMeasurement

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
        candidates: list[CandidateTracks] | list[VowelMeasurement]
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
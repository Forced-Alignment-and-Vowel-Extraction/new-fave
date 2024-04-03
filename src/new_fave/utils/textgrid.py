from aligned_textgrid import AlignedTextGrid, \
    SequenceTier, \
    SequenceInterval
from aligned_textgrid.sequences.tiers import TierGroup

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
    """_summary_

    Args:
        interval (SequenceInterval): _description_

    Returns:
        TierGroup: _description_
    """
    top_tier = get_top_tier(interval)
    return top_tier.within

def get_textgrid(interval: SequenceInterval)->AlignedTextGrid:
    """_summary_

    Args:
        interval (SequenceInterval): _description_

    Returns:
        AlignedTextGrid: _description_
    """
    tier_group = get_tier_group(interval)
    return tier_group.within
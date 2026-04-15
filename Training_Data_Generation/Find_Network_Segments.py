"""
Author: Saskia Knight
Date: January 2026
Description:  Find the GPS times when data is good quality, has no events, and is available for both H1 and L1. 
"""

# System stuff
from __future__ import annotations
import re, os
import requests, json
from typing import Optional, Sequence, Tuple, List

# Basics
import numpy as np
import pandas as pd
import random, math
from bisect import bisect_left

# GW Related
from gwosc.timeline import get_segments
from gwosc.datasets import query_events



Segment = Tuple[int, int]

def intersect_two(a: List[Segment], b: List[Segment], min_len: int = 0) -> List[Segment]:
    """
    Intersection of two segment lists. Keeps only overlapping parts.
    Assumes each list is a list of (start, end) with start < end.
    Works even if boundaries don't line up.
    """
    if not a or not b:
        return []

    a = sorted(a)
    b = sorted(b)

    i = j = 0
    out: List[Segment] = []

    while i < len(a) and j < len(b):
        a0, a1 = a[i]
        b0, b1 = b[j]

        start = max(a0, b0)
        end = min(a1, b1)

        if end > start:
            if end - start >= min_len:
                out.append((start, end))

        # Advance the one that ends first
        if a1 <= b1:
            i += 1
        else:
            j += 1

    return out

def intersect_many(lists: List[List[Segment]], min_len: int = 0) -> List[Segment]:
    """Intersection across many segment lists."""
    # If any list is empty => empty intersection
    if any(not lst for lst in lists):
        return []

    lists = [sorted(lst) for lst in lists]
    result = lists[0]
    for lst in lists[1:]:
        result = intersect_two(result, lst, min_len=0)  # filter at end
        if not result:
            break

    if min_len > 0:
        result = [(s, e) for s, e in result if (e - s) >= min_len]

    return result


def subtract_segments(good: List[Segment], bad: List[Segment]) -> List[Segment]:
    """
    Subtract bad segments from good segments.
    Returns pieces of good that do not overlap bad.
    """
    if not good:
        return []
    if not bad:
        return sorted(good)

    good = sorted(good)
    bad = sorted(bad)

    out: List[Segment] = []
    j = 0

    for gs, ge in good:
        cur = gs

        # advance bad until it might overlap
        while j < len(bad) and bad[j][1] <= cur:
            j += 1

        k = j
        while k < len(bad) and bad[k][0] < ge:
            bs, be = bad[k]

            # if there's a gap before this bad segment, keep it
            if bs > cur:
                out.append((cur, min(bs, ge)))

            # move cur past the bad segment
            if be > cur:
                cur = max(cur, be)

            if cur >= ge:
                break
            k += 1

        if cur < ge:
            out.append((cur, ge))

    return out


def filter_min_len(segs: List[Segment], min_len: int) -> List[Segment]:
    return [(s, e) for (s, e) in segs if (e - s) >= min_len]


def event_pad_seconds(m1, m2, ns_max=2.9):  # 2.9 = max NS mass
    # double check the justification behind these times - maybe plot spectrograms?
    if m1 is None or m2 is None: # Unclassified
        return 2
    elif m1 < ns_max and m2 < ns_max: # BNS
        return 8
    elif (m1 < ns_max) ^ (m2 < ns_max): # NSBH
        return 4
    else:  # BBH
        return 2


GWOSC_EVENT_DETAIL_URL = "https://gwosc.org/eventapi/json/event/{event_name}/v{version}/"
_EVENT_ID_RE = re.compile(r"^(?P<name>.+)-v(?P<ver>\d+)$") # magic DONT TOUCH

def fetch_event_detail(session: requests.Session, start=1368195220, end=1389456018):
    """
    Call GWOSC Event Portal API and return the GPS times and required padding for each event.
    """
    # Find event names within timeframe
    event_list = query_events(select=[f"{end} >= gps-time >= {start}"]) 

    GPS_list = []
    padding_list = []
    for event_id in event_list:

        # Split name into sections for URL building
        m = _EVENT_ID_RE.match(event_id.strip())
        event_name = m.group("name")
        version = int(m.group("ver"))

        url = GWOSC_EVENT_DETAIL_URL.format(event_name=event_name, version=version)

        r = session.get(url)
        r.raise_for_status()
        params = r.json()['events'][event_id]

        GPS = params['GPS']
        mass1 = params['mass_1_source']
        mass2 = params['mass_2_source']
        padding = event_pad_seconds(mass1, mass2)

        GPS_list.append(GPS)
        padding_list.append(padding)

    return GPS_list, padding_list


def get_event_veto_windows(start=1368195220, end=1389456018) -> List[Segment]:
    """
    Build veto windows around known GW event GPS times.
    Query events in [start, end] and convert to GPS.

    """
    session = requests.Session()
    GPS_list, padding_list = fetch_event_detail(session, start, end)

    veto = []
    for t, pad in zip(GPS_list, padding_list):
        veto.append((max(start, t - (0.9*pad)), min(end, t + (0.1*pad))))  # Event is not in the middle
    return veto


def get_science_segments(ifo, start=1368195220, end=1389456018, min_len: int = 68):
    # Data quality problems
    burst2 = get_segments(f"{ifo}_BURST_CAT2", start, end)  # includes burst1 & burst2
    cbc2   = get_segments(f"{ifo}_CBC_CAT2", start, end)    # includes cbc1 & cbc2
    # cw1    = get_segments(f"{ifo}_CW_CAT1", start, end)
    # stoch1 = get_segments(f"{ifo}_STOCH_CAT1", start, end)

    # Hardware injections (these are usually "NO_*" == good times)
    burst_hw_inj   = get_segments(f"{ifo}_NO_BURST_HW_INJ", start, end)
    cbc_hw_inj     = get_segments(f"{ifo}_NO_CBC_HW_INJ", start, end)
    # cw_hw_inj      = get_segments(f"{ifo}_NO_CW_HW_INJ", start, end)
    # detchar_hw_inj = get_segments(f"{ifo}_NO_DETCHAR_HW_INJ", start, end)
    # stoch_hw_inj   = get_segments(f"{ifo}_NO_STOCH_HW_INJ", start, end)

    # Intersection of all GOOD criteria
    segs = intersect_many([burst2, cbc2, 
                        #    cw1, stoch1, 
                           burst_hw_inj, cbc_hw_inj, 
                        #    cw_hw_inj, detchar_hw_inj, stoch_hw_inj
                           ])

    # 2) Remove windows around known GW events
    event_veto = get_event_veto_windows(start, end)
    segs = subtract_segments(segs, event_veto)

    # 3) Drop short leftovers
    segs = filter_min_len(segs, min_len)

    return segs

# Both detectors good
def get_network_science_segments(ifos=("H1", "L1"), start=1368195220, end=1389456018, min_len: int = 68, save=False):
    """
    Return segments where ALL ifos have good-quality data simultaneously.
    """
    per_ifo = [get_science_segments(ifo, start=start, end=end, min_len=min_len) for ifo in ifos]
    # intersect_many expects a list of segment lists
    net = intersect_many(per_ifo)
    net = filter_min_len(net, min_len)

    if save:
        filename = "network_segments_list.json"
        # convert tuples -> lists for JSON compatibility
        net_serialisable = [tuple(seg) for seg in net]

        with open(filename, "w") as f:
            json.dump(net_serialisable, f, indent=2)

    return net
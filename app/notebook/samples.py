"""Realistic sample phase-pick data that exercises the real loaders.

Everything here is generated deterministically (``random.Random`` seed) so the
notebook is reproducible offline.  The generated files are read back through
:mod:`ingest`, so the samples exercise the production code path rather than a
parallel toy implementation.
"""

from __future__ import annotations

import io
import csv
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 20240517
TRUE_VP_VS = 1.732

STATION_POOL: list[tuple[str, str, str]] = [
    ("BRG", "GR", "HHZ"),
    ("MOX", "GR", "HHZ"),
    ("CLL", "GR", "HHZ"),
    ("TANN", "TH", "HHZ"),
    ("WERD", "TH", "HHZ"),
    ("PLN", "SX", "HHZ"),
    ("SCHF", "SX", "HHZ"),
    ("GUNZ", "SX", "HHZ"),
    ("ROHR", "TH", "EHZ"),
    ("NEUB", "TH", "EHZ"),
    ("LAUE", "SX", "EHZ"),
    ("HAIN", "GR", "HHZ"),
    ("GRZ", "GR", "HHZ"),
    ("WIMM", "SX", "EHZ"),
]


@dataclass
class SamplePick:
    station_id: str
    network: str
    channel: str
    p_travel_time: float
    s_minus_p: float


@dataclass
class SampleEvent:
    event_id: str
    origin_time: datetime
    latitude: float
    longitude: float
    depth_m: float
    picks: list[SamplePick]


def sample_events(seed: int = SEED) -> list[SampleEvent]:
    """Four synthetic but scientifically shaped Vogtland-style swarm events."""
    rng = random.Random(seed)
    specs = [
        (
            "gr2024ab12",
            datetime(2024, 5, 17, 4, 12, 9, tzinfo=timezone.utc),
            12,
            0.09,
        ),
        (
            "gr2024ab19",
            datetime(2024, 5, 17, 6, 48, 31, tzinfo=timezone.utc),
            10,
            0.12,
        ),
        (
            "gr2024ac03",
            datetime(2024, 5, 18, 1, 3, 55, tzinfo=timezone.utc),
            9,
            0.07,
        ),
        (
            "gr2024ac27",
            datetime(2024, 5, 19, 22, 37, 2, tzinfo=timezone.utc),
            6,
            0.15,
        ),
    ]
    events: list[SampleEvent] = []
    for event_id, origin_time, n_stations, noise in specs:
        stations = STATION_POOL[:n_stations]
        picks: list[SamplePick] = []
        for index, (station, network, channel) in enumerate(stations):
            p_travel = round(2.4 + 1.45 * index + rng.uniform(-0.25, 0.25), 3)
            s_minus_p = (TRUE_VP_VS - 1.0) * p_travel + rng.gauss(0.0, noise)
            picks.append(
                SamplePick(
                    station, network, channel, p_travel, round(s_minus_p, 3)
                )
            )
        # Deliberate mis-picks so the validation / QC states are visible.
        if len(picks) > 5:
            picks[4].s_minus_p = round(picks[4].s_minus_p + 1.9, 3)  # late S
        if len(picks) > 9:
            picks[9].s_minus_p = round(picks[9].s_minus_p - 1.6, 3)  # early S
        events.append(
            SampleEvent(
                event_id=event_id,
                origin_time=origin_time,
                latitude=round(50.21 + rng.uniform(-0.04, 0.04), 4),
                longitude=round(12.44 + rng.uniform(-0.04, 0.04), 4),
                depth_m=round(rng.uniform(7000.0, 11000.0), 1),
                picks=picks,
            )
        )
    return events


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sample_long_csv(seed: int = SEED) -> str:
    """Long layout: one row per phase pick, absolute timestamps + origin_time."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "event_id",
            "origin_time",
            "network",
            "station_id",
            "channel",
            "phase",
            "time",
        ]
    )
    for event in sample_events(seed):
        for pick in event.picks:
            p_time = event.origin_time + timedelta(seconds=pick.p_travel_time)
            s_time = p_time + timedelta(seconds=pick.s_minus_p)
            for phase, moment in (("P", p_time), ("S", s_time)):
                writer.writerow(
                    [
                        event.event_id,
                        _iso(event.origin_time),
                        pick.network,
                        pick.station_id,
                        pick.channel,
                        phase,
                        _iso(moment),
                    ]
                )
    return buffer.getvalue()


def sample_travel_time_csv(seed: int = SEED) -> str:
    """Wide layout: numeric origin-relative travel times, no absolute times."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "event_id",
            "network",
            "station_id",
            "channel",
            "p_travel_time",
            "s_minus_p",
        ]
    )
    for event in sample_events(seed):
        for pick in event.picks:
            writer.writerow(
                [
                    event.event_id,
                    pick.network,
                    pick.station_id,
                    pick.channel,
                    f"{pick.p_travel_time:.3f}",
                    f"{pick.s_minus_p:.3f}",
                ]
            )
    return buffer.getvalue()


#: Hand-written table where every documented validation code is triggered.
PROBLEM_CSV = """\
event_id,origin_time,network,station_id,channel,phase,time
gr2024bad1,2024-05-20T02:00:00.000Z,GR,BRG,HHZ,P,2024-05-20T02:00:03.100Z
gr2024bad1,2024-05-20T02:00:00.000Z,GR,BRG,HHZ,P,2024-05-20T02:00:03.400Z
gr2024bad1,2024-05-20T02:00:00.000Z,GR,BRG,HHZ,S,2024-05-20T02:00:05.300Z
gr2024bad1,2024-05-20T02:00:00.000Z,GR,MOX,HHZ,P,2024-05-20T02:00:05.900Z
gr2024bad1,2024-05-20T02:00:00.000Z,GR,CLL,HHZ,S,2024-05-20T02:00:11.200Z
gr2024bad1,2024-05-20T02:00:00.000Z,TH,TANN,HHZ,P,2024-05-20T02:00:09.500Z
gr2024bad1,2024-05-20T02:00:00.000Z,TH,TANN,HHZ,S,2024-05-20T02:00:08.900Z
gr2024bad1,2024-05-20T02:00:00.000Z,TH,WERD,HHZ,P,20-05-2024 02:00:11
gr2024bad1,2024-05-20T02:00:00.000Z,TH,WERD,HHZ,S,2024-05-20T02:00:15.400Z
gr2024bad1,2024-05-20T02:00:00.000Z,SX,PLN,HHZ,Lg,2024-05-20T02:00:19.100Z
gr2024bad1,2024-05-20T02:00:00.000Z,SX,SCHF,HHZ,P,2024-05-20T02:00:14.050Z
gr2024bad1,2024-05-20T02:00:00.000Z,SX,SCHF,HHZ,S,2024-05-20T02:00:20.180Z
gr2024bad2,,GR,BRG,HHZ,P,2024-05-20T04:10:02.900Z
gr2024bad2,,GR,BRG,HHZ,S,2024-05-20T04:10:05.100Z
gr2024bad2,,GR,MOX,HHZ,P,2024-05-20T04:10:04.400Z
gr2024bad2,,GR,MOX,HHZ,S,2024-05-20T04:10:07.600Z
"""


def write_sample_csvs(directory: str | Path) -> dict[str, Path]:
    """Write the three sample CSV tables and return their paths."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written = {
        "long": target / "sample_picks_long.csv",
        "travel_time": target / "sample_picks_travel_time.csv",
        "problems": target / "sample_picks_problems.csv",
    }
    written["long"].write_text(sample_long_csv(), encoding="utf-8")
    written["travel_time"].write_text(
        sample_travel_time_csv(), encoding="utf-8"
    )
    written["problems"].write_text(PROBLEM_CSV, encoding="utf-8")
    return written


def build_sample_catalog(seed: int = SEED, include_problems: bool = True):
    """Build an ObsPy ``Catalog`` from the sample events.

    With ``include_problems`` the catalog also carries an event whose origin has
    no time and a station with a duplicated P pick, so the QuakeML loader's
    validation paths are exercised too.
    """
    from obspy import UTCDateTime
    from obspy.core.event import (
        Arrival,
        Catalog,
        Event,
        Origin,
        Pick,
        WaveformStreamID,
    )

    catalog = Catalog()
    events = sample_events(seed)
    for index, sample in enumerate(events):
        event = Event(resource_id=f"smi:local/event/{sample.event_id}")
        origin_time = UTCDateTime(sample.origin_time)
        origin = Origin(
            resource_id=f"smi:local/origin/{sample.event_id}",
            time=origin_time,
            latitude=sample.latitude,
            longitude=sample.longitude,
            depth=sample.depth_m,
        )
        arrivals: list[Arrival] = []
        for pick_no, sample_pick in enumerate(sample.picks, start=1):
            waveform = WaveformStreamID(
                network_code=sample_pick.network,
                station_code=sample_pick.station_id,
                channel_code=sample_pick.channel,
            )
            offsets = {
                "P": sample_pick.p_travel_time,
                "S": sample_pick.p_travel_time + sample_pick.s_minus_p,
            }
            for phase, offset in offsets.items():
                pick = Pick(
                    resource_id=(
                        f"smi:local/pick/{sample.event_id}/"
                        f"{sample_pick.station_id}/{phase}"
                    ),
                    time=origin_time + offset,
                    waveform_id=waveform,
                    phase_hint=phase,
                    evaluation_mode="manual",
                )
                event.picks.append(pick)
                arrivals.append(Arrival(pick_id=pick.resource_id, phase=phase))
            if include_problems and index == 0 and pick_no == 2:
                # duplicated P pick, 0.2 s later
                event.picks.append(
                    Pick(
                        resource_id=(
                            f"smi:local/pick/{sample.event_id}/"
                            f"{sample_pick.station_id}/P-dup"
                        ),
                        time=origin_time + sample_pick.p_travel_time + 0.2,
                        waveform_id=waveform,
                        phase_hint="P",
                        evaluation_mode="automatic",
                    )
                )
        origin.arrivals = arrivals
        event.origins = [origin]
        event.preferred_origin_id = origin.resource_id
        catalog.append(event)

    if include_problems:
        # An event whose only origin carries no time at all.
        sample = events[0]
        broken = Event(resource_id="smi:local/event/gr2024noorigin")
        origin = Origin(resource_id="smi:local/origin/gr2024noorigin")
        reference = UTCDateTime(sample.origin_time)
        for sample_pick in sample.picks[:5]:
            waveform = WaveformStreamID(
                network_code=sample_pick.network,
                station_code=sample_pick.station_id,
                channel_code=sample_pick.channel,
            )
            for phase, offset in (
                ("P", sample_pick.p_travel_time),
                ("S", sample_pick.p_travel_time + sample_pick.s_minus_p),
            ):
                broken.picks.append(
                    Pick(
                        resource_id=(
                            f"smi:local/pick/gr2024noorigin/"
                            f"{sample_pick.station_id}/{phase}"
                        ),
                        time=reference + offset,
                        waveform_id=waveform,
                        phase_hint=phase,
                    )
                )
        broken.origins = [origin]
        catalog.append(broken)
    return catalog


def write_sample_quakeml(
    path: str | Path,
    seed: int = SEED,
    include_problems: bool = True,
) -> Path:
    """Write a programmatically generated sample QuakeML file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        catalog = build_sample_catalog(
            seed=seed, include_problems=include_problems
        )
        catalog.write(target, format="QUAKEML")
    except Exception as e:
        logging.exception(f"Error: {e}")
        raise
    return target

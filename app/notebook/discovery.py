"""Real FDSN discovery of LO stations and Hispaniola-region earthquakes.

Geographic scope
----------------
**Hispaniola** is the Caribbean island shared by the **Dominican Republic** and
**Haiti**.  Network **LO** is the *Observatorio Sismológico Politécnico Loyola*
(**OSPL**) network, operated in the **Dominican Republic**; its metadata is
served by **EARTHSCOPE** (formerly IRIS).  Regional earthquake catalogs are
queried from **USGS**, which has good coverage of the Caribbean.

The default bounding box is deliberately broad, so it covers the whole of
Hispaniola (both countries) *and* the nearby offshore seismicity along the
Septentrional / Enriquillo fault systems, the Puerto Rico trench approaches and
the Muertos trough:

    latitude   16.5 N .. 20.5 N
    longitude  75.5 W .. 67.0 W   (i.e. -75.5 .. -67.0)

Design contract
---------------
* Nothing in this module performs network access at import time.  Every query
  is an explicit function call, which the notebook wires to a button.
* Every failure mode returns a :class:`DiscoveryResult` carrying actionable
  messages instead of raising: no data, malformed date/magnitude range,
  provider failure (bad request / outage) and offline/DNS/timeout errors.
* Results are tidy: a list of flat dicts with stable IDs, ready for pandas or a
  Panel ``MultiChoice`` / ``MultiSelect``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

#: Default station metadata provider for network LO (OSPL, Dominican Republic).
STATION_PROVIDER = "EARTHSCOPE"

#: Default event catalog provider for the Hispaniola region.
EVENT_PROVIDER = "USGS"

#: Network code of the Observatorio Sismológico Politécnico Loyola (OSPL).
LO_NETWORK = "LO"

LO_NETWORK_DESCRIPTION = (
    "LO — Observatorio Sismológico Politécnico Loyola (OSPL), "
    "Dominican Republic, on the island of Hispaniola "
    "(shared by the Dominican Republic and Haiti)."
)

#: Documented default bounding box: all of Hispaniola plus nearby offshore
#: seismicity.  (min_lat, max_lat, min_lon, max_lon) in degrees.
HISPANIOLA_BBOX: tuple[float, float, float, float] = (16.5, 20.5, -75.5, -67.0)

BBOX_DOC = """\
DEFAULT REGION — HISPANIOLA AND SURROUNDINGS
--------------------------------------------
  latitude   16.5 N .. 20.5 N
  longitude  -75.5 .. -67.0  (75.5 W .. 67.0 W)

Hispaniola is shared by the Dominican Republic and Haiti.  The box is wider
than the island itself so that offshore events along the Septentrional and
Enriquillo fault zones, the Muertos trough and the western approaches to the
Puerto Rico trench are included.

PROVIDERS
---------
  stations  EARTHSCOPE  (the only FDSN node that serves network LO)
  events    USGS        (regional Caribbean catalog)

Network LO is the Observatorio Sismológico Politécnico Loyola (OSPL) of the
Dominican Republic.
"""

STATION_COLUMNS: list[str] = [
    "station_uid",
    "network",
    "station_id",
    "site",
    "latitude",
    "longitude",
    "elevation_m",
    "start_date",
    "end_date",
    "channels",
]

EVENT_COLUMNS: list[str] = [
    "event_uid",
    "event_id",
    "origin_time",
    "latitude",
    "longitude",
    "depth_km",
    "magnitude",
    "magnitude_type",
    "event_type",
    "region",
]

#: Magnitude threshold used when the notebook control is left at its default.
DEFAULT_MIN_MAGNITUDE = 3.0

#: Default UTC window length (days) offered by the notebook controls.
DEFAULT_WINDOW_DAYS = 365


@dataclass
class DiscoveryMessage:
    """One actionable message from a discovery query."""

    code: str
    level: str  # "error" | "warning" | "info"
    message: str

    def __str__(self) -> str:
        prefix = {"error": "ERROR  ", "warning": "warning", "info": "info   "}
        return f"[{prefix.get(self.level, self.level)}] {self.code}: {self.message}"


@dataclass
class DiscoveryResult:
    """Tidy rows plus every message raised while querying a provider."""

    kind: str  # "stations" | "events"
    provider: str
    query: dict[str, str] = field(default_factory=dict)
    rows: list[dict[str, str | float | int]] = field(default_factory=list)
    messages: list[DiscoveryMessage] = field(default_factory=list)

    @property
    def errors(self) -> list[DiscoveryMessage]:
        return [m for m in self.messages if m.level == "error"]

    @property
    def warnings(self) -> list[DiscoveryMessage]:
        return [m for m in self.messages if m.level == "warning"]

    @property
    def ok(self) -> bool:
        return bool(self.rows) and not self.errors

    @property
    def columns(self) -> list[str]:
        return STATION_COLUMNS if self.kind == "stations" else EVENT_COLUMNS

    def ids(self) -> list[str]:
        """Stable IDs, suitable as ``MultiSelect`` / ``MultiChoice`` values."""
        key = "station_uid" if self.kind == "stations" else "event_uid"
        return [str(row[key]) for row in self.rows]

    def options(self) -> dict[str, str]:
        """``{label: value}`` mapping for Panel selection widgets."""
        if self.kind == "stations":
            return {
                f"{row['station_uid']} — {row['site'] or 'unnamed site'}"
                f"  ({float(row['latitude']):.3f}, {float(row['longitude']):.3f})": str(
                    row["station_uid"]
                )
                for row in self.rows
            }
        return {
            f"{row['origin_time']}  M{float(row['magnitude']):.1f} "
            f"{row['magnitude_type']}  z={float(row['depth_km']):.1f} km "
            f"({float(row['latitude']):.2f}, {float(row['longitude']):.2f})": str(
                row["event_uid"]
            )
            for row in self.rows
        }

    def select(
        self, uids: list[str] | tuple[str, ...]
    ) -> list[dict[str, str | float | int]]:
        """Rows whose stable ID is in ``uids``, in table order."""
        wanted = set(uids)
        key = "station_uid" if self.kind == "stations" else "event_uid"
        return [row for row in self.rows if str(row[key]) in wanted]

    def report(self) -> str:
        lines = [
            f"kind              {self.kind}",
            f"provider          {self.provider}",
            f"rows              {len(self.rows)}",
            f"errors/warnings   {len(self.errors)} / {len(self.warnings)}",
        ]
        if self.query:
            lines.append(
                "query             "
                + ", ".join(f"{k}={v}" for k, v in self.query.items())
            )
        if self.messages:
            lines.append("")
            lines.extend(str(m) for m in self.messages)
        return "\n".join(lines)

    def to_dataframe(self):  # pragma: no cover - notebook convenience
        import pandas as pd

        return pd.DataFrame(self.rows, columns=self.columns)


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _fail(
    kind: str, provider: str, code: str, message: str, query: dict
) -> DiscoveryResult:
    return DiscoveryResult(
        kind=kind,
        provider=provider,
        query=query,
        messages=[DiscoveryMessage(code, "error", message)],
    )


def _utc(value: object):
    """Coerce anything date-like to ``obspy.UTCDateTime`` (raises ValueError)."""
    from obspy import UTCDateTime

    if value is None:
        raise ValueError("a UTC date is required")
    try:
        return UTCDateTime(value)
    except Exception as exc:  # obspy raises TypeError/ValueError variants
        logging.exception("Unexpected error")
        raise ValueError(
            f"{value!r} is not a UTC date/time — use e.g. '2024-01-01' or "
            "'2024-01-01T00:00:00'"
        ) from exc


def validate_window(
    starttime: object, endtime: object
) -> tuple[object, object]:
    """Validate a UTC range, raising ``ValueError`` with an actionable message."""
    start = _utc(starttime)
    end = _utc(endtime)
    if end <= start:
        raise ValueError(
            f"the end of the window ({end}) is not after its start ({start}); "
            "swap the two dates or widen the range"
        )
    return start, end


def validate_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Validate a (min_lat, max_lat, min_lon, max_lon) box in degrees."""
    try:
        min_lat, max_lat, min_lon, max_lon = (float(v) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "bbox must be four numbers: (min_lat, max_lat, min_lon, max_lon)"
        ) from exc
    if not (-90.0 <= min_lat < max_lat <= 90.0):
        raise ValueError(
            f"latitudes must satisfy -90 <= min < max <= 90 (got {min_lat}, {max_lat})"
        )
    if not (-180.0 <= min_lon < max_lon <= 180.0):
        raise ValueError(
            f"longitudes must satisfy -180 <= min < max <= 180 (got {min_lon}, {max_lon}); "
            "west longitudes are negative, so Hispaniola is -75.5 .. -67.0"
        )
    return min_lat, max_lat, min_lon, max_lon


def _classify(exc: Exception) -> tuple[str, str]:
    """Map an ObsPy/urllib failure onto (code, actionable message)."""
    name = type(exc).__name__
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else name
    if name == "FDSNNoDataException":
        return "no_data", (
            "the provider returned no data for this query (HTTP 204) — widen the "
            "date range, lower the magnitude threshold, or check the region"
        )
    if name == "FDSNBadRequestException":
        return "bad_request", (
            f"the provider rejected the query as invalid (HTTP 400): {text}. "
            "Check the network code, the date range and the bounding box"
        )
    if name == "FDSNNoServiceException":
        return "provider_unreachable", (
            f"no FDSN service could be discovered at this provider: {text}. "
            "The node may be down, or this machine is offline"
        )
    if name.startswith("FDSN"):
        return "provider_failure", (
            f"the FDSN provider failed: {text}. Retry, or try again later"
        )
    if any(
        k in name
        for k in ("URLError", "Timeout", "Socket", "Connection", "gaierror")
    ):
        return "offline", (
            f"the request could not reach the network ({text}) — this notebook "
            "needs internet access for FDSN discovery. Work offline with the "
            "sample CSV / QuakeML files instead"
        )
    return "query_failed", f"unexpected {name}: {text}"


# ---------------------------------------------------------------------------
# station discovery (EARTHSCOPE / network LO / OSPL, Dominican Republic)
# ---------------------------------------------------------------------------
def fetch_stations(
    network: str = LO_NETWORK,
    starttime: object = None,
    endtime: object = None,
    bbox: tuple[float, float, float, float] | None = HISPANIOLA_BBOX,
    provider: str = STATION_PROVIDER,
    level: str = "channel",
) -> DiscoveryResult:
    """Query station metadata for one network (default ``LO``, the OSPL network).

    Never raises: validation and provider problems come back as error messages
    on the returned :class:`DiscoveryResult`.
    """
    query: dict[str, str] = {"network": network, "level": level}
    try:
        from obspy import UTCDateTime
        from obspy.clients.fdsn import Client
    except Exception as e:
        logging.exception(f"Error: {e}")
        return _fail(
            "stations",
            provider,
            "obspy_missing",
            f"ObsPy is required for FDSN discovery ({e}); install obspy>=1.5",
            query,
        )

    kwargs: dict[str, object] = {"network": network, "level": level}
    try:
        if starttime is not None or endtime is not None:
            start, end = validate_window(
                starttime if starttime is not None else UTCDateTime(0),
                endtime if endtime is not None else UTCDateTime(),
            )
            kwargs["starttime"] = start
            kwargs["endtime"] = end
            query["window"] = f"{start} .. {end}"
        if bbox is not None:
            min_lat, max_lat, min_lon, max_lon = validate_bbox(bbox)
            kwargs.update(
                minlatitude=min_lat,
                maxlatitude=max_lat,
                minlongitude=min_lon,
                maxlongitude=max_lon,
            )
            query["bbox"] = f"{min_lat}/{max_lat}/{min_lon}/{max_lon}"
    except ValueError as e:
        return _fail("stations", provider, "malformed_range", e, query)

    try:
        inventory = Client(provider).get_stations(**kwargs)
    except Exception as e:
        logging.exception(f"Error: {e}")
        code, message = _classify(e)
        return _fail("stations", provider, code, message, query)

    rows: list[dict[str, str | float | int]] = []
    for net in inventory:
        for sta in net:
            uid = f"{net.code}.{sta.code}"
            channels = (
                sorted({ch.code for ch in sta.channels}) if sta.channels else []
            )
            rows.append(
                {
                    "station_uid": uid,
                    "network": str(net.code),
                    "station_id": str(sta.code),
                    "site": str(sta.site.name)
                    if sta.site and sta.site.name
                    else "",
                    "latitude": float(sta.latitude),
                    "longitude": float(sta.longitude),
                    "elevation_m": float(sta.elevation or 0.0),
                    "start_date": str(sta.start_date) if sta.start_date else "",
                    "end_date": str(sta.end_date) if sta.end_date else "",
                    "channels": ",".join(channels),
                }
            )
    rows.sort(key=lambda r: str(r["station_uid"]))

    result = DiscoveryResult("stations", provider, query, rows)
    if not rows:
        result.messages.append(
            DiscoveryMessage(
                "no_data",
                "error",
                f"network {network!r} returned no stations inside the requested box and "
                "window — widen the region or the dates. Note that LO (OSPL, Dominican "
                f"Republic) is served by {STATION_PROVIDER} only",
            )
        )
    else:
        result.messages.append(
            DiscoveryMessage(
                "ok",
                "info",
                f"{len(rows)} station(s) from {provider}: {LO_NETWORK_DESCRIPTION}"
                if network == LO_NETWORK
                else f"{len(rows)} station(s) from {provider}",
            )
        )
    return result


def fetch_lo_stations(**kwargs) -> DiscoveryResult:
    """Convenience wrapper: network LO (OSPL, Dominican Republic) via EARTHSCOPE."""
    kwargs.setdefault("network", LO_NETWORK)
    kwargs.setdefault("provider", STATION_PROVIDER)
    return fetch_stations(**kwargs)


# ---------------------------------------------------------------------------
# event discovery (USGS / Hispaniola and surroundings)
# ---------------------------------------------------------------------------
def fetch_events(
    starttime: object,
    endtime: object,
    minmagnitude: float = DEFAULT_MIN_MAGNITUDE,
    bbox: tuple[float, float, float, float] = HISPANIOLA_BBOX,
    provider: str = EVENT_PROVIDER,
    limit: int = 200,
    orderby: str = "time",
) -> DiscoveryResult:
    """Query earthquakes across Hispaniola (Dominican Republic + Haiti) and offshore.

    Never raises: validation and provider problems come back as error messages
    on the returned :class:`DiscoveryResult`.
    """
    query: dict[str, str] = {
        "minmagnitude": f"{minmagnitude}",
        "limit": str(limit),
    }
    try:
        from obspy.clients.fdsn import Client
    except Exception as e:
        logging.exception(f"Error: {e}")
        return _fail(
            "events",
            provider,
            "obspy_missing",
            f"ObsPy is required for FDSN discovery ({e}); install obspy>=1.5",
            query,
        )

    try:
        start, end = validate_window(starttime, endtime)
        min_lat, max_lat, min_lon, max_lon = validate_bbox(bbox)
        magnitude = float(minmagnitude)
        if magnitude < -1.0 or magnitude > 10.0:
            raise ValueError(
                f"minimum magnitude {magnitude} is outside a plausible range "
                "(-1 .. 10); the regional default is 3.0"
            )
        if int(limit) < 1:
            raise ValueError("limit must be at least 1")
    except (ValueError, TypeError) as e:
        return _fail("events", provider, "malformed_range", e, query)

    query["window"] = f"{start} .. {end}"
    query["bbox"] = f"{min_lat}/{max_lat}/{min_lon}/{max_lon}"

    try:
        catalog = Client(provider).get_events(
            starttime=start,
            endtime=end,
            minlatitude=min_lat,
            maxlatitude=max_lat,
            minlongitude=min_lon,
            maxlongitude=max_lon,
            minmagnitude=magnitude,
            limit=int(limit),
            orderby=orderby,
        )
    except Exception as e:
        logging.exception(f"Error: {e}")
        code, message = _classify(e)
        return _fail("events", provider, code, message, query)

    rows: list[dict[str, str | float | int]] = []
    messages: list[DiscoveryMessage] = []
    for index, event in enumerate(catalog, start=1):
        origin = None
        try:
            origin = event.preferred_origin()
        except Exception:  # pragma: no cover - defensive
            logging.exception("Unexpected error")
        if origin is None or origin.time is None:
            origin = next(
                (o for o in event.origins if o.time is not None), None
            )
        if origin is None:
            messages.append(
                DiscoveryMessage(
                    "missing_origin",
                    "warning",
                    f"event {index} carries no origin with a time and is skipped",
                )
            )
            continue
        magnitude_obj = None
        try:
            magnitude_obj = event.preferred_magnitude()
        except Exception:  # pragma: no cover - defensive
            logging.exception("Unexpected error")
        if magnitude_obj is None and event.magnitudes:
            magnitude_obj = event.magnitudes[0]
        raw_id = str(event.resource_id)
        short_id = raw_id.rsplit("eventid=", 1)[-1].split("&", 1)[0]
        short_id = short_id.rsplit("/", 1)[-1] or f"event-{index}"
        region = ""
        if event.event_descriptions:
            region = str(event.event_descriptions[0].text or "")
        rows.append(
            {
                "event_uid": short_id,
                "event_id": raw_id,
                "origin_time": str(origin.time),
                "latitude": float(origin.latitude),
                "longitude": float(origin.longitude),
                "depth_km": round(float(origin.depth or 0.0) / 1000.0, 2),
                "magnitude": float(magnitude_obj.mag)
                if magnitude_obj and magnitude_obj.mag is not None
                else float("nan"),
                "magnitude_type": str(
                    getattr(magnitude_obj, "magnitude_type", "") or ""
                ),
                "event_type": str(event.event_type or ""),
                "region": region,
            }
        )
    rows.sort(key=lambda r: str(r["origin_time"]), reverse=True)

    result = DiscoveryResult("events", provider, query, rows, messages)
    if not rows:
        result.messages.append(
            DiscoveryMessage(
                "no_data",
                "error",
                f"no events of M>={magnitude} between {start} and {end} inside "
                f"{min_lat}/{max_lat} N, {min_lon}/{max_lon} — lower the magnitude "
                "threshold or widen the UTC window",
            )
        )
    else:
        result.messages.append(
            DiscoveryMessage(
                "ok",
                "info",
                f"{len(rows)} event(s) from {provider} across Hispaniola "
                "(Dominican Republic and Haiti) and its offshore surroundings",
            )
        )
    return result


def fetch_hispaniola_events(**kwargs) -> DiscoveryResult:
    """Convenience wrapper: USGS events inside the documented Hispaniola box."""
    kwargs.setdefault("provider", EVENT_PROVIDER)
    kwargs.setdefault("bbox", HISPANIOLA_BBOX)
    return fetch_events(**kwargs)


def default_window(days: int = DEFAULT_WINDOW_DAYS) -> tuple[object, object]:
    """A (start, end) UTC pair ending now — no network access involved."""
    from obspy import UTCDateTime

    end = UTCDateTime()
    return end - int(days) * 86400, end

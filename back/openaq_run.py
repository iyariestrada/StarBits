# -*- coding: utf-8 -*-
# openaq_run.py
#
# pip install requests pandas python-dateutil pytz

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta, timezone
import os, requests, pandas as pd
from typing import Dict, Iterable, List, Tuple, Optional
from pathlib import Path
import json, os

# 1) Tu archivo de secretos local (NO lo subas al repo)
SECRETS_PATH = Path(r"C:\Users\Admin\Downloads\Nasa\tempo_secrets.json")

def _load_secret_from_file(path: Path, key: str) -> str | None:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    except Exception:
        pass
    return None

# 2) Prioridad: ENV > secrets.json
API_KEY = os.getenv("openaq_api_key") or _load_secret_from_file(SECRETS_PATH, "openaq_api_key")

if not API_KEY:
    raise SystemExit(
        "Falta OPENAQ_API_KEY. Define la variable de entorno o guarda en tempo_secrets.json:\n"
        '{ "OPENAQ_API_KEY": "tu_api_key" }'
    )

BASE = "https://api.openaq.org/v3"
HEAD = {"X-API-Key": API_KEY}

# --- Helpers de metadata de sensores (para lat/lon) ---
def _get_sensor_meta(sensor_id: int) -> dict:
    """Devuelve metadata del sensor (incluye coordinates y parameter)."""
    url = f"{BASE}/sensors/{sensor_id}"
    r = requests.get(url, headers=HEAD, timeout=30)
    r.raise_for_status()
    return r.json().get("results", {})

def _meta_df_for_sensors(sensor_ids: list[int]) -> pd.DataFrame:
    metas = []
    for sid in sensor_ids:
        try:
            m = _get_sensor_meta(int(sid))
            if m:
                metas.append(m)
        except Exception:
            continue
    if not metas:
        return pd.DataFrame()
    dfm = pd.json_normalize(metas).rename(columns={
        "id": "sensor_id",
        "coordinates.latitude": "lat",
        "coordinates.longitude": "lon",
        "parameter.name": "parameter_meta",
        "parameter.units": "unit_meta",
        "location.id": "location_id",
        "name": "sensor_name",
    })
    if "sensor_id" in dfm.columns:
        dfm["sensor_id"] = pd.to_numeric(dfm["sensor_id"], errors="coerce").astype("Int64")
    return dfm


# ----- Tabla que me diste (zona -> lista de (sensor_id, soporta)) -----
# soporta: "pm25", "pm10" o "ambos"
ZONE_OPENAQ_SENSORS: Dict[int, List[Tuple[int, str]]] = {
    1: [(8759, "pm25"), (793, "ambos"), (895, "ambos"), (663, "ambos"), (1948, "ambos")],
    2: [(1404, "ambos"), (1866, "ambos"), (1199, "ambos"), (1189, "ambos"), (825, "ambos")],
    3: [(2272, "ambos"), (8213, "ambos"), (288, "ambos"), (2743106, "ambos"), (3376346, "ambos")],
    4: [(509, "pm25"), (542, "ambos"), (1177, "ambos"), (5638300, "ambos"), (11570, "ambos")],
    5: [(8735, "pm25"), (813, "ambos"), (5941, "pm25"), (4454899, "ambos"), (7973, "ambos")],
    6: [(1381, "ambos"), (2197, "ambos"), (1620, "pm10"), (1342, "pm25"), (7941, "pm25")],
    7: [(236033, "pm25"), (1535910, "ambos"), (226149, "ambos"), (922, "pm25"), (1377907, "ambos")],
    8: [(326608, "pm25"), (8799, "pm25"), (2119, "ambos"), (3239941, "ambos"), (8476, "ambos")],
    9: [(4219738, "pm25"), (1590, "ambos"), (2161, "ambos"), (494922, "ambos"), (9600, "ambos")],
    10: [(10675, "ambos"), (2037, "pm25"), (1679245, "pm25"), (774, "pm25"), (345, "ambos")],
}

def _now_utc_rounded():
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

def _isoZ(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def _supports(target: str, flag: str) -> bool:
    if flag == "ambos":
        return True
    return flag.lower() == target.lower()

def _get_hourly_by_sensor(sensor_id: int, hours: int = 6) -> List[dict]:
    end_utc = _now_utc_rounded()
    start_utc = end_utc - timedelta(hours=hours)
    params = {
        "date_from": _isoZ(start_utc),
        "date_to": _isoZ(end_utc),
        "limit": 1000,  # suficiente para 6h/1h
    }
    url = f"{BASE}/sensors/{sensor_id}/measurements/hourly"
    r = requests.get(url, headers=HEAD, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Campos típicos de measurements/hourly
    # Ejemplos esperados:
    # - datetime.utc
    # - value
    # - unit
    # - parameter.name (pm25/pm10)
    # - coordinates.{latitude,longitude} (a veces en hourly no viene; depende del backend)
    rename_map = {
        "datetime.utc": "datetime_utc",
        "datetime.local": "datetime_local",
        "parameter.name": "parameter",
        "coordinates.latitude": "latitude",
        "coordinates.longitude": "longitude",
        "sensor.id": "sensor_id",
        "location.id": "location_id",
    }
    out = df.rename(columns=rename_map)
    # Asegura columnas clave
    for c in ["datetime_utc", "parameter", "value", "unit", "sensor_id"]:
        if c not in out.columns:
            out[c] = pd.NA
    return out

def run_openaq_for_zone(
    zone_id: int,
    out_dir: str,
    *,
    hours: int = 6,
    want_pm25: bool = True,
    want_pm10: bool = True,
) -> dict:
    """
    Descarga últimas `hours` horas para los sensores listados en ZONE_OPENAQ_SENSORS[zone_id].
    Guarda CSVs separados por especie (pm25/pm10) y un CSV con el último valor por sensor/especie.
    Devuelve dict con estatus y rutas.
    """
    sensors = ZONE_OPENAQ_SENSORS.get(zone_id, [])
    if not sensors:
        return {"ok": False, "reason": f"No hay sensores OpenAQ registrados para zona {zone_id}."}

    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)

    rows_all: List[pd.DataFrame] = []
    errors: List[str] = []
    for sid, flag in sensors:
        try:
            data = _get_hourly_by_sensor(int(sid), hours=hours)
            if not data:
                continue
            df = pd.json_normalize(data)
            if df.empty:
                continue
            df = _normalize_df(df)
            df["zone_id"] = zone_id
            df["sensor_id"] = int(sid)
            rows_all.append(df)
        except Exception as ex:
            errors.append(f"[sensor {sid}] {ex}")

    if not rows_all:
        return {"ok": False, "reason": "No hubo datos en la ventana para esos sensores.", "errors": errors}

    df_all = pd.concat(rows_all, ignore_index=True)

    # --- merge coordinates por sensor (para poder graficar grilla) ---
    unique_sids = sorted({int(x) for x in df_all["sensor_id"].dropna().unique()})
    df_meta = _meta_df_for_sensors(unique_sids)

    if not df_meta.empty:
        # usa solo las columnas que existan en df_meta
        wanted = ["sensor_id", "lat", "lon", "sensor_name"]
        cols = [c for c in wanted if c in df_meta.columns]
        if "sensor_id" in cols:
            df_all = df_all.merge(df_meta[cols], on="sensor_id", how="left")

    # estandariza lat/lon si vienen como latitude/longitude en el hourly
    if "lat" not in df_all.columns and "latitude" in df_all.columns:
        df_all = df_all.rename(columns={"latitude": "lat"})
    if "lon" not in df_all.columns and "longitude" in df_all.columns:
        df_all = df_all.rename(columns={"longitude": "lon"})


    # Filtra por especie
    outs = {"ok": True, "zone": zone_id, "errors": errors}

    def _save_species(sp_name: str, label: str):
        sp = df_all[df_all["parameter"].str.lower() == sp_name]
        if sp.empty:
            outs[label] = {"ok": False, "rows": 0, "csv": None}
            return
        # Aplica capacidades por sensor
        valid_sids = [sid for sid, flag in sensors if _supports(sp_name, flag)]
        sp = sp[sp["sensor_id"].isin(valid_sids)]
        if sp.empty:
            outs[label] = {"ok": False, "rows": 0, "csv": None}
            return

        csv_path = out_path / f"zone{zone_id}_{sp_name}_hourly.csv"
        sp.to_csv(csv_path, index=False)
        outs[label] = {"ok": True, "rows": int(len(sp)), "csv": str(csv_path)}

        # promedio 6h por sensor
        sp["datetime_utc"] = pd.to_datetime(sp["datetime_utc"], errors="coerce", utc=True)
        sp = sp.dropna(subset=["datetime_utc", "value"])
        sp["value"] = pd.to_numeric(sp["value"], errors="coerce")

        per_sensor = sp.groupby("sensor_id")["value"].mean().rename("mean_6h").reset_index()
        # adjunta lat/lon del sensor para poder mapear centros (por si quieres grilla de sensores)
        if "lat" in sp.columns and "lon" in sp.columns:
            per_sensor = per_sensor.merge(
                sp.groupby("sensor_id")[["lat", "lon"]].agg("first").reset_index(),
                on="sensor_id",
                how="left"
            )

        summary_csv = out_path / f"zone{zone_id}_{sp_name}_hourly_summary.csv"
        per_sensor.to_csv(summary_csv, index=False)

        outs[label]["summary_csv"] = str(summary_csv)
        outs[label]["mean6h_all"] = float(per_sensor["mean_6h"].mean(skipna=True)) if not per_sensor.empty else None


    if want_pm25:
        _save_species("pm25", "PM25")
    if want_pm10:
        _save_species("pm10", "PM10")

    # Además, un “latest” combinado por especie (solo si existen)
    latest_frames = []
    for key in ("PM25", "PM10"):
        info = outs.get(key, {})
        if info.get("latest_csv"):
            latest_frames.append(pd.read_csv(info["latest_csv"]))
    if latest_frames:
        latest_all = pd.concat(latest_frames, ignore_index=True)
        latest_all_csv = out_path / f"zone{zone_id}_openaq_latest.csv"
        latest_all.to_csv(latest_all_csv, index=False)
        outs["LATEST_ALL"] = {"csv": str(latest_all_csv), "rows": int(len(latest_all))}

    return outs



def _meta_df_for_sensors(sensor_ids: list[int]) -> pd.DataFrame:
    metas = []
    for sid in sensor_ids:
        try:
            m = _get_sensor_meta(int(sid))
            if m:
                metas.append(m)
        except Exception:
            # si falla uno, seguimos con los demás
            continue
    if not metas:
        return pd.DataFrame()
    dfm = pd.json_normalize(metas).rename(columns={
        "id": "sensor_id",
        "coordinates.latitude": "lat",
        "coordinates.longitude": "lon",
        "parameter.name": "parameter_meta",
        "parameter.units": "unit_meta",
        "location.id": "location_id",
        "name": "sensor_name"
    })
    # Asegura tipos
    if "sensor_id" in dfm.columns:
        dfm["sensor_id"] = pd.to_numeric(dfm["sensor_id"], errors="coerce").astype("Int64")
    return dfm

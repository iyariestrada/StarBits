# -*- coding: utf-8 -*-
# zone_run.py
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json, numpy as np, pandas as pd, xarray as xr
import earthaccess as ea
from matplotlib.path import Path as MplPath

from zones_points import ZONE_POINTS  # <— nuevo

HCHO_FIXED_WINDOW = ("2025-09-16T18:03:08Z", "2025-09-16T21:02:51Z")
AI_FIXED_WINDOWS = [
    ("2023-09-20T08:00:00Z", "2023-09-20T14:00:00Z"),
    ("2023-09-21T08:00:00Z", "2023-09-21T14:00:00Z"),
]
CONCEPT_O3TOT_L2_V04 = "C3685912131-LARC_CLOUD"

def safe_engine():
    try:
        import netCDF4  # noqa
        return None
    except ImportError:
        return "h5netcdf"
_ENGINE = safe_engine()

# ===== helpers (idénticos a tu zone3_run) =====
def poly_bbox_latlon(points_latlon):
    lats = [p[0] for p in points_latlon]; lons = [p[1] for p in points_latlon]
    return (min(lons), min(lats), max(lons), max(lats))

def point_in_poly(lon, lat, poly_latlon):
    x, y = lon, lat; inside = False; n = len(poly_latlon)
    for i in range(n):
        lat_i, lon_i = poly_latlon[i]; lat_j, lon_j = poly_latlon[(i+1) % n]
        xi, yi = lon_i, lat_i; xj, yj = lon_j, lat_j
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect: inside = not inside
    return inside

def mask_points_in_poly(lons, lats, poly_latlon):
    verts = [(lon, lat) for (lat, lon) in poly_latlon]
    path = MplPath(verts, closed=True)
    P = np.column_stack([lons.astype(float), lats.astype(float)])
    return path.contains_points(P)

def to_temporal(end_utc_str: str, window_hours: int):
    end_dt = datetime.strptime(end_utc_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(hours=window_hours)
    return (start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))

def time_to_isoZ_strings(x):
    arr = x.values.ravel()
    if np.issubdtype(arr.dtype, np.number):
        dt = pd.to_datetime(arr.astype("float64"), unit="s", utc=True, origin="1980-01-06 00:00:00+00:00")
    else:
        dt = pd.to_datetime(arr, utc=False, errors="coerce")
        dt = dt.tz_localize("UTC") if getattr(dt, "tz", None) is None else dt.tz_convert("UTC")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def align_to(lat, var):
    if var is None: return None
    td = lat.dims
    if set(var.dims) == set(td):
        return var.transpose(*td) if var.dims != td else var
    if set(var.dims).issubset(set(td)):
        var_b, _ = xr.broadcast(var, lat)
        return var_b.transpose(*td)
    raise RuntimeError("Dimensiones incompatibles")

def search_and_download(concept_id, bbox, temporal, dest):
    gs = ea.search_data(concept_id=concept_id, temporal=temporal, bounding_box=bbox)
    if not gs: return []
    try:    files = ea.download(gs, local_path=str(dest))
    except TypeError: files = ea.download(gs, destination=str(dest))
    return files or []

# ======== factor común: nombres por zona ========
def _base_name(zone_id: int, stem: str) -> str:
    return f"zone{zone_id}_{stem}"

# ======== Procesos (idénticos, pero con zone_points y base dinámico) ========
# Copia tus funciones _process_no2/_process_hcho/_process_ai/_process_ai_from_o3tot/_process_o3
# cambiando SOLO las líneas de guardado: base = dest_base / _base_name(zone_id, "no2")  (etc)
# y recibiendo zone_id y zone_points como parámetros.

# Ejemplo con NO2 (los demás análogos):
def _process_no2(zone_id, zone_points, temporal, dest_base: Path, logs: list):
    import xarray as xr
    CONCEPT_NO2 = "C3685668972-LARC_CLOUD"
    bbox = poly_bbox_latlon(zone_points)
    files = search_and_download(CONCEPT_NO2, bbox, temporal, dest_base)
    if not files:
        logs.append("[NO2] Sin granulos en la ventana/bbox."); return pd.DataFrame()

    dfs = []
    for f in files:
        p = Path(f); nc = p
        ds_geo = xr.open_dataset(str(nc), group="geolocation", engine=_ENGINE)
        ds_prod= xr.open_dataset(str(nc), group="product",     engine=_ENGINE)
        ds_sup = xr.open_dataset(str(nc), group="support_data",engine=_ENGINE)

        LAT = ds_geo["latitude"]; LON = ds_geo["longitude"]; TIME = ds_geo["time"]
        NO2 = ds_prod["vertical_column_troposphere"]; MDQF= ds_prod["main_data_quality_flag"]; ECF = ds_sup["eff_cloud_fraction"]
        NO2U= ds_prod.get("vertical_column_troposphere_uncertainty")
        def A(x): return align_to(LAT, x) if x is not None else None
        LON, TIME, NO2, MDQF, ECF, NO2U = map(A, (LON, TIME, NO2, MDQF, ECF, NO2U))

        qa_mask = (ECF <= 0.1) & (MDQF == 0); NO2 = NO2.where(qa_mask)
        t_iso = time_to_isoZ_strings(TIME)
        df = pd.DataFrame({
            "species":"NO2", "time_utc": t_iso,
            "lat": LAT.values.ravel(), "lon": LON.values.ravel(),
            "value": NO2.values.ravel(),
            "eff_cloud_fraction": ECF.values.ravel(),
            "main_data_quality_flag": MDQF.values.ravel(),
            "granule_filename": p.name
        })
        if NO2U is not None: df["uncertainty"] = NO2U.values.ravel()
        df = df[(df["eff_cloud_fraction"] <= 0.1) & (df["main_data_quality_flag"] == 0)]
        if not df.empty:
            mask = mask_points_in_poly(df["lon"].to_numpy(), df["lat"].to_numpy(), zone_points)
            df = df[mask]
            if not df.empty: dfs.append(df)

        ds_geo.close(); ds_prod.close(); ds_sup.close()

    if not dfs:
        logs.append("[NO2] No hubo datos válidos dentro del polígono tras QA."); return pd.DataFrame()

    res = pd.concat(dfs, ignore_index=True)
    base = dest_base / _base_name(zone_id, "no2")
    res.to_csv(base.with_suffix(".csv"), index=False)
    logs.append(f"[NO2] OK | filas={len(res)} | {base.with_suffix('.csv').name}")
    return res

# Reutiliza tus funciones _process_hcho/_process_ai/_process_ai_from_o3tot/_process_o3
# tal cual, cambiando base = dest_base / _base_name(zone_id, "<stem>") y pasando zone_id/zone_points.

# ======== Run wrappers (ACEPTAN zone_id) ========
def run_no2_for_zone(zone_id: int, END_UTC_STR: str, WINDOW_HOURS: int, out_dir: str):
    logs = []; out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    temporal = to_temporal(END_UTC_STR, WINDOW_HOURS)
    logs.append(f"Ventana UTC: {temporal[0]} → {temporal[1]}")
    for strat in ("environment","netrc","interactive"):
        try: ea.login(strategy=strat); break
        except Exception: continue

    zone_points = ZONE_POINTS.get(zone_id)
    if not zone_points: return {"ok": False, "csv_no2": None, "rows": 0, "logs": logs + [f"Zona {zone_id} no registrada."]}

    df_no2 = _process_no2(zone_id, zone_points, temporal, out_path, logs)
    if df_no2.empty: return {"ok": False, "csv_no2": None, "rows": 0, "logs": logs}
    csv_no2 = str((out_path / _base_name(zone_id, "no2")).with_suffix(".csv"))
    return {"ok": True, "csv_no2": csv_no2, "rows": len(df_no2), "logs": logs}

def _save_summary(df: pd.DataFrame, out_base: Path, species: str):
    # Crea "<nombre>_summary.csv" junto a out_base, sin usar with_suffix
    out_csv = out_base.parent / f"{out_base.name}_summary.csv"

    # Si el DF viene con 'value_DU' (caso O3 antes de renombrar), lo adapto
    if "value" not in df.columns and "value_DU" in df.columns:
        df = df.rename(columns={"value_DU": "value"})

    if df.empty or "value" not in df.columns:
        pd.DataFrame([{"species": species, "n": 0, "mean_value": float("nan")}]).to_csv(out_csv, index=False)
        return str(out_csv)

    s = pd.DataFrame([{
        "species": species,
        "n": int(len(df)),
        "mean_value": float(np.nanmean(pd.to_numeric(df["value"], errors="coerce").to_numpy()))
    }])
    s.to_csv(out_csv, index=False)
    return str(out_csv)

def _find_concept(keyword: str, provider: str = "LARC_CLOUD"):
    try:
        cols = ea.search_datasets(keyword=keyword, provider=provider)
    except Exception:
        cols = []
    if not cols:
        return None
    # escoge la primera (o ajusta criterio si quieres)
    return cols[0]["meta"]["concept-id"]

def _process_o3(zone_id, zone_points, temporal, dest_base: Path, logs: list):
    """
    Extrae O3 (Total Column Ozone) preferentemente de L2 O3TOT V04 (Provisional).
    Si no hay L2 en la ventana/bbox, intenta L3 O3TOT.
    Filtros recomendados: quality_flag==0, SZA<=80, VZA<=80, fc<0.5.
    """
    bbox = poly_bbox_latlon(zone_points)

    # --- L2: TEMPO_O3TOT_L2_V04 (Provisional) ---
    concept_l2 = CONCEPT_O3TOT_L2_V04
    files = search_and_download(concept_l2, bbox, temporal, dest_base) if concept_l2 else []
    dfs = []

    if files:
        for f in files:
            p = Path(f); nc = p
            try:
                ds_geo = xr.open_dataset(str(nc), group="geolocation", engine=_ENGINE)
                ds_prod= xr.open_dataset(str(nc), group="product",     engine=_ENGINE)
            except Exception:
                logs.append(f"[O3] Estructura inesperada (L2) en {p.name}.")
                continue

            LAT = ds_geo.get("latitude")
            LON = ds_geo.get("longitude")
            TIME= ds_geo.get("time")
            SZA = ds_geo.get("solar_zenith_angle")
            VZA = ds_geo.get("viewing_zenith_angle")

            O3  = ds_prod.get("column_amount_o3")  # DU
            CF  = ds_prod.get("fc")                # effective cloud fraction
            QF  = ds_prod.get("quality_flag")      # 0 = good
            UAI = ds_prod.get("uv_aerosol_index")  # opcional

            if (LAT is None) or (LON is None) or (O3 is None):
                logs.append(f"[O3] Faltan variables clave en {p.name}.")
                for d in (ds_geo, ds_prod):
                    try: d.close()
                    except: pass
                continue

            def A(x): return align_to(LAT, x) if x is not None else None
            LON = A(LON); TIME = A(TIME); O3 = A(O3)
            CF  = A(CF);  QF   = A(QF);   UAI= A(UAI)
            SZA = A(SZA); VZA  = A(VZA)

            mask = xr.ones_like(O3, dtype=bool)
            if QF  is not None: mask = mask & (QF == 0)
            if CF  is not None: mask = mask & (CF < 0.5)
            if SZA is not None: mask = mask & (SZA <= 80.0)
            if VZA is not None: mask = mask & (VZA <= 80.0)

            O3m  = O3.where(mask);  LONm = LON.where(mask); LATm = LAT.where(mask)
            t_iso = time_to_isoZ_strings(TIME) if TIME is not None else pd.Series([""]*O3m.size)

            df = pd.DataFrame({
                "species": "O3",
                "time_utc": t_iso,
                "lat": LATm.values.ravel(),
                "lon": LONm.values.ravel(),
                "value_DU": O3m.values.ravel(),
                "granule_filename": p.name
            })
            if CF  is not None: df["cloud_fraction"] = CF.values.ravel()
            if QF  is not None: df["quality_flag"]   = QF.values.ravel()
            if UAI is not None: df["uv_aerosol_index"] = UAI.values.ravel()
            if SZA is not None: df["sza_deg"] = SZA.values.ravel()
            if VZA is not None: df["vza_deg"] = VZA.values.ravel()

            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat","lon","value_DU"])
            if not df.empty:
                mpoly = mask_points_in_poly(df["lon"].to_numpy(), df["lat"].to_numpy(), zone_points)
                df = df[mpoly]
                if not df.empty:
                    dfs.append(df)

            ds_geo.close(); ds_prod.close()

        if dfs:
            res = pd.concat(dfs, ignore_index=True)
            base = dest_base / _base_name(zone_id, "o3")
            res.to_csv(base.with_suffix(".csv"), index=False)
            sum_path = _save_summary(res.rename(columns={"value_DU":"value"}), base, "O3")
            logs.append(f"[O3] OK L2 | filas={len(res)} | {base.with_suffix('.csv').name} | resumen={Path(sum_path).name}")
            return res
        else:
            logs.append("[O3] No hubo datos L2 válidos dentro del polígono tras QA.")

    # --- L3 fallback ---
    concept_l3 = _find_concept("TEMPO O3TOT L3 V04") or _find_concept("TEMPO O3TOT L3")
    files3 = search_and_download(concept_l3, bbox, temporal, dest_base) if concept_l3 else []
    if not files3:
        logs.append("[O3] Sin granulos L3 en la ventana/bbox.")
        return pd.DataFrame()

    dfs = []
    for f in files3:
        p = Path(f); nc = p
        try:
            ds = xr.open_dataset(str(nc), engine=_ENGINE)
        except Exception:
            logs.append(f"[O3] No pude abrir L3 {p.name}")
            continue

        var = None
        for nm in ["column_amount_o3", "ozone_total_column", "ozone"]:
            if nm in ds.variables:
                var = ds[nm]; break

        lat_name = next((n for n in ds.variables if n.lower() in ("lat","latitude")), None)
        lon_name = next((n for n in ds.variables if n.lower() in ("lon","longitude")), None)
        if (var is None) or (lat_name is None) or (lon_name is None):
            ds.close(); continue

        LAT = ds[lat_name]; LON = ds[lon_name]
        if LAT.ndim == 1 and LON.ndim == 1:
            lon2, lat2 = np.meshgrid(LON.values, LAT.values)
            vals = np.asarray(var.values)
        else:
            lat2 = np.asarray(LAT.values); lon2 = np.asarray(LON.values)
            vals = np.asarray(var.values)

        flat = pd.DataFrame({
            "species":"O3",
            "lat": lat2.ravel(),
            "lon": lon2.ravel(),
            "value_DU": vals.ravel(),
            "granule_filename": p.name
        }).replace([np.inf, -np.inf], np.nan).dropna(subset=["lat","lon","value_DU"])

        if not flat.empty:
            mpoly = mask_points_in_poly(flat["lon"].to_numpy(), flat["lat"].to_numpy(), zone_points)
            flat = flat[mpoly]
            if not flat.empty:
                dfs.append(flat)

        ds.close()

    if not dfs:
        logs.append("[O3] No hubo datos L3 dentro del polígono.")
        return pd.DataFrame()

    res = pd.concat(dfs, ignore_index=True)
    base = dest_base / _base_name(zone_id, "o3")
    res.to_csv(base.with_suffix(".csv"), index=False)
    sum_path = _save_summary(res.rename(columns={"value_DU":"value"}), base, "O3")
    logs.append(f"[O3] OK L3 | filas={len(res)} | {base.with_suffix('.csv').name} | resumen={Path(sum_path).name}")
    return res

def _temporal_list_from_fixed_or_main(fixed_list, temporal_main):
    """
    Devuelve una lista de ventanas [(startZ,endZ), ...].
    Si fixed_list es no vacía → la usa; de lo contrario usa [temporal_main].
    """
    if fixed_list and len(fixed_list) > 0:
        return fixed_list
    return [temporal_main]

def _process_ai_from_o3tot(zone_id, zone_points, temporal_main, dest_base: Path, logs: list, *, fixed_windows=None):
    """
    Extrae AI (UV aerosol index) desde O3TOT (L2 preferente, L3 fallback).
    Guarda zone{zone_id}_ai.csv y zone{zone_id}_ai_summary.csv.
    """
    temporal_list = _temporal_list_from_fixed_or_main(fixed_windows, temporal_main)
    dfs_all = []

    for temporal in temporal_list:
        bbox = poly_bbox_latlon(zone_points)

        # --- L2 primero ---
        concept_l2 = _find_concept("TEMPO O3TOT L2 V03") or _find_concept("TEMPO O3TOT L2")
        files = search_and_download(concept_l2, bbox, temporal, dest_base) if concept_l2 else []
        got = False

        if files:
            for f in files:
                p = Path(f); nc = p
                try:
                    ds_geo = xr.open_dataset(str(nc), group="geolocation", engine=_ENGINE)
                    ds_prod= xr.open_dataset(str(nc), group="product",     engine=_ENGINE)
                except Exception:
                    logs.append(f"[AI] Estructura inesperada (L2/O3TOT) en {p.name}.")
                    continue

                LAT = ds_geo.get("latitude");  LON = ds_geo.get("longitude"); TIME = ds_geo.get("time")
                SZA = ds_geo.get("solar_zenith_angle"); VZA = ds_geo.get("viewing_zenith_angle")
                AI  = ds_prod.get("uv_aerosol_index")
                O3Q = ds_prod.get("quality_flag")
                CF  = ds_prod.get("fc")

                if (LAT is None) or (LON is None) or (AI is None):
                    ds_geo.close(); ds_prod.close()
                    continue

                def A(x): return align_to(LAT, x) if x is not None else None
                LON = A(LON); TIME = A(TIME); AI = A(AI); CF = A(CF); O3Q = A(O3Q); SZA = A(SZA); VZA = A(VZA)

                mask = xr.ones_like(AI, dtype=bool)
                if O3Q is not None: mask = mask & (O3Q == 0)
                if CF  is not None: mask = mask & (CF < 0.5)
                if SZA is not None: mask = mask & (SZA <= 80.0)
                if VZA is not None: mask = mask & (VZA <= 80.0)

                AIm = AI.where(mask); LONm = LON.where(mask); LATm = LAT.where(mask)
                t_iso = time_to_isoZ_strings(TIME) if TIME is not None else pd.Series([""]*AIm.size)

                df = pd.DataFrame({
                    "species": "AI",
                    "time_utc": t_iso,
                    "lat": LATm.values.ravel(),
                    "lon": LONm.values.ravel(),
                    "value": AIm.values.ravel(),
                    "granule_filename": p.name
                }).replace([np.inf, -np.inf], np.nan).dropna(subset=["lat","lon","value"])

                if not df.empty:
                    mpoly = mask_points_in_poly(df["lon"].to_numpy(), df["lat"].to_numpy(), zone_points)
                    df = df[mpoly]
                    if not df.empty:
                        dfs_all.append(df); got = True

                ds_geo.close(); ds_prod.close()

        # --- L3 si L2 no dejó nada ---
        if not got:
            concept_l3 = _find_concept("TEMPO O3TOT L3 V03") or _find_concept("TEMPO O3TOT L3")
            files3 = search_and_download(concept_l3, bbox, temporal, dest_base) if concept_l3 else []
            for f in files3:
                p = Path(f); nc = p
                try:
                    ds = xr.open_dataset(str(nc), engine=_ENGINE)
                except Exception:
                    logs.append(f"[AI] No pude abrir L3 {p.name}")
                    continue

                var = None
                for nm in ["uv_aerosol_index", "aerosol_index", "absorbing_aerosol_index"]:
                    if nm in ds.variables:
                        var = ds[nm]; break

                lat_name = next((n for n in ds.variables if n.lower() in ("lat","latitude")), None)
                lon_name = next((n for n in ds.variables if n.lower() in ("lon","longitude")), None

                )
                if (var is None) or (lat_name is None) or (lon_name is None):
                    ds.close(); continue

                LAT = ds[lat_name]; LON = ds[lon_name]
                if LAT.ndim == 1 and LON.ndim == 1:
                    lon2, lat2 = np.meshgrid(LON.values, LAT.values)
                    vals = np.asarray(var.values)
                else:
                    lat2 = np.asarray(LAT.values); lon2 = np.asarray(LON.values)
                    vals = np.asarray(var.values)

                flat = pd.DataFrame({
                    "species": "AI",
                    "lat": lat2.ravel(),
                    "lon": lon2.ravel(),
                    "value": vals.ravel(),
                    "granule_filename": p.name
                }).replace([np.inf, -np.inf], np.nan).dropna(subset=["lat","lon","value"])

                if not flat.empty:
                    mpoly = mask_points_in_poly(flat["lon"].to_numpy(), flat["lat"].to_numpy(), zone_points)
                    flat = flat[mpoly]
                    if not flat.empty:
                        dfs_all.append(flat)

                ds.close()

    if not dfs_all:
        logs.append("[AI] No hubo datos válidos dentro del polígono en las ventanas indicadas.")
        return pd.DataFrame()

    res = pd.concat(dfs_all, ignore_index=True)
    base = dest_base / _base_name(zone_id, "ai")
    res.to_csv(base.with_suffix(".csv"), index=False)
    sum_path = _save_summary(res, base, "AI")
    logs.append(f"[AI] OK | filas={len(res)} | {base.with_suffix('.csv').name} | resumen={Path(sum_path).name}")
    return res


def _find_concept_hcho_v03(provider: str = "LARC_CLOUD"):
    """
    Devuelve el concept-id de TEMPO_HCHO_L2_V03 en LARC_CLOUD.
    Si 'short_name' no está disponible en tu earthaccess, cae a keyword y filtra.
    """
    try:
        # Intento 1: por short_name exacto (lo más confiable)
        cols = ea.search_datasets(short_name="TEMPO_HCHO_L2_V03", provider=provider)
        if not cols:
            # Intento 2: por keyword y filtrando por el patrón L2_V03
            cols = ea.search_datasets(keyword="TEMPO HCHO L2 V03", provider=provider)
            cols = [c for c in cols if "_L2_V03" in c.get("umm", {}).get("ShortName", "")]
    except Exception:
        cols = []
    return cols[0]["meta"]["concept-id"] if cols else None

def _process_hcho(zone_id, zone_points, temporal, dest_base: Path, logs: list, *, temporal_override=None):
    # Decide la ventana temporal a usar (override > constante > la que viene de main)
    temporal_hcho = temporal_override or HCHO_FIXED_WINDOW or temporal
    if temporal_hcho != temporal:
        logs.append(f"[HCHO] Usando ventana fija: {temporal_hcho[0]} → {temporal_hcho[1]}")

    concept = _find_concept_hcho_v03()
    if not concept:
        logs.append("[HCHO] No encontré TEMPO_HCHO_L2_V03 en LARC_CLOUD.")
        return pd.DataFrame()

    bbox = poly_bbox_latlon(zone_points)
    files = search_and_download(concept, bbox, temporal_hcho, dest_base)
    if not files:
        logs.append("[HCHO] Sin granulos en la ventana/bbox.")
        return pd.DataFrame()

    dfs = []
    for f in files:
        p = Path(f); nc = p
        try:
            ds_geo = xr.open_dataset(str(nc), group="geolocation", engine=_ENGINE)
            ds_prod= xr.open_dataset(str(nc), group="product",     engine=_ENGINE)
            ds_sup = xr.open_dataset(str(nc), group="support_data",engine=_ENGINE)
        except Exception:
            logs.append(f"[HCHO] Estructura inesperada en {p.name}.")
            continue

        LAT = ds_geo["latitude"]; LON = ds_geo["longitude"]; TIME = ds_geo["time"]

        var = None
        for nm in ["formaldehyde_vertical_column", "formaldehyde_column", "column_amount_hcho"]:
            if nm in ds_prod.variables:
                var = ds_prod[nm]; break
        if var is None:
            logs.append(f"[HCHO] Variable principal no encontrada en {p.name}.")
            ds_geo.close(); ds_prod.close(); ds_sup.close(); continue

        ECF = ds_sup.get("eff_cloud_fraction")
        MDQF= ds_prod.get("main_data_quality_flag")
        UNC = None
        for nm in ["formaldehyde_vertical_column_uncertainty", "formaldehyde_column_uncertainty"]:
            if nm in ds_prod.variables:
                UNC = ds_prod[nm]; break

        def A(x): return align_to(LAT, x) if x is not None else None
        LON, TIME, var, ECF, MDQF, UNC = map(A, (LON, TIME, var, ECF, MDQF, UNC))

        # QA
        if (ECF is not None) and (MDQF is not None):
            qa_mask = (ECF <= 0.2) & (MDQF == 0)
            var = var.where(qa_mask); LON = LON.where(qa_mask)

        t_iso = time_to_isoZ_strings(TIME)
        df = pd.DataFrame({
            "species": "HCHO",
            "time_utc": t_iso,
            "lat": LAT.values.ravel(),
            "lon": LON.values.ravel(),
            "value": var.values.ravel(),
            "granule_filename": p.name
        })
        if ECF is not None:  df["eff_cloud_fraction"] = ECF.values.ravel()
        if MDQF is not None: df["main_data_quality_flag"] = MDQF.values.ravel()
        if UNC is not None:  df["uncertainty"] = UNC.values.ravel()

        if "eff_cloud_fraction" in df.columns:
            df = df[df["eff_cloud_fraction"] <= 0.2]
        if "main_data_quality_flag" in df.columns:
            df = df[df["main_data_quality_flag"] == 0]

        if not df.empty:
            mask = mask_points_in_poly(df["lon"].to_numpy(), df["lat"].to_numpy(), zone_points)
            df = df[mask]

        if not df.empty:
            dfs.append(df)

        ds_geo.close(); ds_prod.close(); ds_sup.close()

    if not dfs:
        logs.append("[HCHO] No hubo datos válidos dentro del polígono.")
        return pd.DataFrame()

    res = pd.concat(dfs, ignore_index=True)
    base = dest_base / _base_name(zone_id, "hcho")
    res.to_csv(base.with_suffix(".csv"), index=False)
    sum_path = _save_summary(res, base, "HCHO")
    logs.append(f"[HCHO] OK | filas={len(res)} | {base.with_suffix('.csv').name} | resumen={Path(sum_path).name}")
    return res


def run_all_for_zone(zone_id: int, END_UTC_STR: str, WINDOW_HOURS: int, out_dir: str):
    out = {}
    for strat in ("environment","netrc","interactive"):
        try: ea.login(strategy=strat); break
        except Exception: continue

    temporal = to_temporal(END_UTC_STR, WINDOW_HOURS)
    logs = [f"Ventana UTC: {temporal[0]} → {temporal[1]}"]
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)

    zone_points = ZONE_POINTS.get(zone_id)
    if not zone_points:
        logs.append(f"Zona {zone_id} no registrada en ZONE_POINTS.")
        return {"summary_json": None, "logs": logs}

    # Usa tus implementaciones refactorizadas (aquí muestro sólo NO2; llama las otras igual):
    df_no2 = _process_no2(zone_id, zone_points, temporal, out_path, logs)
    csv_no2 = str((out_path / _base_name(zone_id, "no2")).with_suffix(".csv")) if not df_no2.empty else None
    out["NO2"] = {"ok": not df_no2.empty, "csv": csv_no2, "rows": int(len(df_no2))}

    df_o3 = _process_o3(zone_id, zone_points, temporal, out_path, logs)  # <- tu versión refactor
    df_ai = _process_ai_from_o3tot(zone_id, zone_points, temporal, out_path, logs, fixed_windows=AI_FIXED_WINDOWS)
    df_hcho = _process_hcho(zone_id, zone_points, temporal, out_path, logs, temporal_override=(HCHO_FIXED_WINDOW or temporal))

    csv_o3  = str((out_path / _base_name(zone_id, "o3")).with_suffix(".csv"))   if not df_o3.empty else None
    csv_ai  = str((out_path / _base_name(zone_id, "ai")).with_suffix(".csv"))   if not df_ai.empty else None
    csv_hcho= str((out_path / _base_name(zone_id, "hcho")).with_suffix(".csv")) if not df_hcho.empty else None

    out["O3"] =  {"ok": not df_o3.empty,   "csv": csv_o3,   "rows": int(len(df_o3))}
    out["AI"] =  {"ok": not df_ai.empty,   "csv": csv_ai,   "rows": int(len(df_ai))}
    out["HCHO"]={"ok": not df_hcho.empty, "csv": csv_hcho, "rows": int(len(df_hcho))}

    # JSON resumen (idéntico pero generalizando a zone_id y nombres):
    summary = {
        "zone": zone_id,
        "windows": {"main": {"start": temporal[0], "end": temporal[1]}},
        "species": {
            "NO2": {"csv": csv_no2, "rows": out["NO2"]["rows"], "mean": None},
             "O3":  {"csv": csv_o3,   "rows": out["O3"]["rows"],  "mean": None},
             "AI":  {"csv": csv_ai,   "rows": out["AI"]["rows"],  "mean": None},
             "HCHO":{"csv": csv_hcho, "rows": out["HCHO"]["rows"],"mean": None},
        }
    }
    out_json = out_path / f"zone{zone_id}_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logs.append(f"[SUMMARY] Escribí {out_json.name}")

    out["summary_json"] = str(out_json)
    out["logs"] = logs
    return out

# (Opcional) Si necesitas mesh_and_broadcast u otras utilidades, copia aquí
# tus implementaciones actuales de zone3_run.py sin cambios.
# === PLOTTER GENÉRICO: grilla 2D del valor promedio ===
import matplotlib.pyplot as plt

def _get_value_column(df: pd.DataFrame):
    if "value" in df.columns:
        return "value"
    if "value_DU" in df.columns:
        return "value_DU"
    # fallback: intenta detectar alguna columna numérica razonable
    candidates = [c for c in df.columns if c.lower() in ("no2", "o3", "ai", "hcho")]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("No encontré columna de valores (value o value_DU) en el CSV.")

    """
    Dibuja una grilla promediando por celdas. Si no pasas csv_path,
    busca: <out_dir>/zone{zone_id}_{species_lower}.csv
    Devuelve la ruta del PNG.
    """
    sp = species.strip().lower()
    out_path = Path(out_dir)
    if csv_path is None:
        csv_path = out_path / f"zone{zone_id}_{sp}.csv"
    else:
        csv_path = Path(csv_path)

    if zone_points is None:
        try:
            from zones_points import ZONE_POINTS
            zone_points = ZONE_POINTS.get(zone_id)
        except Exception:
            zone_points = None

    if not csv_path.exists():
        raise FileNotFoundError(f"No existe CSV para {species} en {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV {csv_path} está vacío.")

    # Selección de columnas
    val_col = _get_value_column(df)
    # Limpieza básica
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon", val_col])

    # Ajuste de límites: usa el bbox del polígono si lo tienes; si no, usa data
    if zone_points:
        minlon, minlat, maxlon, maxlat = poly_bbox_latlon(zone_points)
    else:
        minlat, maxlat = float(df["lat"].min()), float(df["lat"].max())
        minlon, maxlon = float(df["lon"].min()), float(df["lon"].max())

    # Construir bordes de bins
    nbins_lat = max(1, int(np.ceil((maxlat - minlat) / binsize_deg)))
    nbins_lon = max(1, int(np.ceil((maxlon - minlon) / binsize_deg)))
    lat_edges = np.linspace(minlat, maxlat, nbins_lat + 1)
    lon_edges = np.linspace(minlon, maxlon, nbins_lon + 1)

    # Promedio por celda con histogram2d (pesos)
    vals = pd.to_numeric(df[val_col], errors="coerce").to_numpy()
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()

    # suma de valores y conteo por celda
    H_sum, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges], weights=vals)
    H_cnt, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        H_mean = H_sum / H_cnt
    H_mean = np.where(H_cnt > 0, H_mean, np.nan)

    # Escala de color opcional (clip por cuantiles si no se pasa vmin/vmax)
    if vmin is None or vmax is None:
        v = H_mean[np.isfinite(H_mean)]
        if v.size > 0:
            q1, q99 = np.nanpercentile(v, [1, 99])
            vmin = q1 if vmin is None else vmin
            vmax = q99 if vmax is None else vmax

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    # extent = (xmin, xmax, ymin, ymax)  -> ojo: lon primero, luego lat
    extent = (lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1])
    im = ax.imshow(
        H_mean.T,             # imshow espera [Y,X]; trasponemos para (lat,lon)
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    units = "DU" if species.upper() == "O3" and val_col == "value_DU" else "col. units"
    cbar.set_label(f"{species.upper()} [{units}]")

    ax.set_title(f"Zona {zone_id} • {species.upper()} • Grid {binsize_deg}°")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.2, linestyle=":")

    # contorno del polígono
    if zone_points:
        poly_lats = [p[0] for p in zone_points]
        poly_lons = [p[1] for p in zone_points]
        ax.plot(poly_lons + [poly_lons[0]], poly_lats + [poly_lats[0]], lw=1.2)

    png_path = out_path / f"zone{zone_id}_{sp}_grid.png"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)

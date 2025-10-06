# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers comunes ----------
def _read_points(csv_path, *, coerce_value_du=False):
    df = pd.read_csv(csv_path)
    # normaliza nombres
    if "latitude" in df.columns and "lat" not in df.columns:
        df = df.rename(columns={"latitude": "lat"})
    if "longitude" in df.columns and "lon" not in df.columns:
        df = df.rename(columns={"longitude": "lon"})
    if coerce_value_du and "value" not in df.columns and "value_DU" in df.columns:
        df = df.rename(columns={"value_DU": "value"})
    return df

def _percentile_limits(arr, pmin=2, pmax=98):
    a = np.asarray(arr)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None, None
    return np.nanpercentile(a, [pmin, pmax])

def _grid_mean(df_pts, binsize_deg=0.25):
    lon = df_pts["lon"].to_numpy()
    lat = df_pts["lat"].to_numpy()
    val = df_pts["value"].to_numpy()
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    if lon_max == lon_min: lon_max += binsize_deg
    if lat_max == lat_min: lat_max += binsize_deg
    lon_edges = np.arange(lon_min, lon_max + binsize_deg, binsize_deg)
    lat_edges = np.arange(lat_min, lat_max + binsize_deg, binsize_deg)
    nx, ny = len(lon_edges)-1, len(lat_edges)-1
    sumg = np.zeros((ny,nx)); cntg = np.zeros((ny,nx), dtype=int)
    ix = np.digitize(lon, lon_edges) - 1
    iy = np.digitize(lat, lat_edges) - 1
    ok = (ix>=0)&(ix<nx)&(iy>=0)&(iy<ny)&np.isfinite(val)
    for x,y,v in zip(ix[ok], iy[ok], val[ok]):
        sumg[y,x] += v; cntg[y,x] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_grid = sumg / cntg
    mean_grid[cntg==0] = np.nan
    extent = [lon_edges.min(), lon_edges.max(), lat_edges.min(), lat_edges.max()]
    return mean_grid, extent, int(ok.sum())

def _save_imshow(mean_grid, extent, title, cbar_label, out_png, *, pclip=(2,98), dpi=150):
    vmin, vmax = _percentile_limits(mean_grid, *pclip)
    fig, ax = plt.subplots(figsize=(10,8), dpi=dpi)
    im = ax.imshow(mean_grid, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax); cbar.set_label(cbar_label)
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)
    return str(out_png)

def _pm_has_latlon(csv_path: str) -> bool:
    try:
        df = pd.read_csv(csv_path, nrows=5)
        cols = {c.lower() for c in df.columns}
        return ("lat" in cols and "lon" in cols) or ("latitude" in cols and "longitude" in cols)
    except Exception:
        return False

def _series_promedio_por_hora(csv_path: str, out_png: Path, titulo: str, ylabel: str):
    df = pd.read_csv(csv_path)
    if df.empty or "datetime_utc" not in df.columns:
        return None
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime_utc"])
    g = df.groupby("datetime_utc")["value"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8,4), dpi=150)
    ax.plot(g["datetime_utc"], g["value"], marker="o", lw=1.5)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_title(titulo); ax.set_xlabel("UTC"); ax.set_ylabel(ylabel)
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)
    return str(out_png)

def _mean_from_csv(csv_path, value_col="value"):
    try:
        df = pd.read_csv(csv_path)
        if df.empty or value_col not in df.columns:
            return None
        return float(pd.to_numeric(df[value_col], errors="coerce").mean(skipna=True))
    except Exception:
        return None

# ---------- NO2 ----------
def plot_no2(zone_id, csv_path, out_dir, *, binsize_deg=0.25, pclip=(2,98)):
    df = _read_points(csv_path)
    if not {"lon","lat","value"}.issubset(df.columns):
        return {"ok": False, "msg": "NO2: faltan columnas lon/lat/value"}
    # filtros suaves (por si llegaron columnas)
    if "main_data_quality_flag" in df: df = df[df["main_data_quality_flag"]==0]
    if "eff_cloud_fraction" in df:     df = df[df["eff_cloud_fraction"]<=0.1]
    pts = df[["lon","lat","value"]].dropna()
    if pts.empty:
        return {"ok": False, "msg": "NO2: sin puntos válidos"}
    grid, extent, n = _grid_mean(pts, binsize_deg=binsize_deg)
    out_png = Path(out_dir)/f"zone{zone_id}_no2_grid.png"
    png = _save_imshow(grid, extent, f"Zona {zone_id} • NO₂ (rejilla {binsize_deg}°)", "NO₂ (mol/m²)", out_png, pclip=pclip)
    return {"ok": True, "png": png, "n": n, "mean": float(np.nanmean(pts["value"]))}

# ---------- HCHO ----------
def plot_hcho(zone_id, csv_path, out_dir, *, binsize_deg=0.25, pclip=(2,98)):
    df = _read_points(csv_path)
    if not {"lon","lat","value"}.issubset(df.columns):
        return {"ok": False, "msg": "HCHO: faltan lon/lat/value"}
    pts = df[["lon","lat","value"]].dropna()
    if pts.empty:
        return {"ok": False, "msg": "HCHO: sin puntos"}
    grid, extent, n = _grid_mean(pts, binsize_deg=binsize_deg)
    out_png = Path(out_dir)/f"zone{zone_id}_hcho_grid.png"
    png = _save_imshow(grid, extent, f"Zona {zone_id} • HCHO (rejilla {binsize_deg}°)", "HCHO (mol/m²)", out_png, pclip=pclip)
    return {"ok": True, "png": png, "n": n, "mean": float(np.nanmean(pts["value"]))}

# ---------- AI ----------
def plot_ai(zone_id, csv_path, out_dir, *, binsize_deg=0.25, pclip=(2,98)):
    df = _read_points(csv_path)
    if not {"lon","lat","value"}.issubset(df.columns):
        return {"ok": False, "msg": "AI: faltan lon/lat/value"}
    pts = df[["lon","lat","value"]].dropna()
    if pts.empty:
        return {"ok": False, "msg": "AI: sin puntos"}
    grid, extent, n = _grid_mean(pts, binsize_deg=binsize_deg)
    out_png = Path(out_dir)/f"zone{zone_id}_ai_grid.png"
    png = _save_imshow(grid, extent, f"Zona {zone_id} • Aerosol Index (rejilla {binsize_deg}°)", "AI (adim.)", out_png, pclip=pclip)
    return {"ok": True, "png": png, "n": n, "mean": float(np.nanmean(pts["value"]))}

# ---------- O3 (usa DU) ----------
def plot_o3(zone_id, csv_path, out_dir, *, binsize_deg=0.2, pclip=(5,95)):
    df = _read_points(csv_path, coerce_value_du=True)  # value_DU -> value
    if not {"lon","lat","value"}.issubset(df.columns):
        return {"ok": False, "msg": "O3: faltan lon/lat/value(_DU)"}
    pts = df[["lon","lat","value"]].dropna()
    if pts.empty:
        return {"ok": False, "msg": "O3: sin puntos"}
    grid, extent, n = _grid_mean(pts, binsize_deg=binsize_deg)
    out_png = Path(out_dir)/f"zone{zone_id}_o3_grid.png"
    png = _save_imshow(grid, extent, f"Zona {zone_id} • O₃ total (rejilla {binsize_deg}°)", "O₃ (DU)", out_png, pclip=pclip)
    return {"ok": True, "png": png, "n": n, "mean": float(np.nanmean(pts["value"]))}

# ---------- PM (2.5 / 10) ----------
def plot_pm(zone_id, csv_path, out_dir, *, species_name: str):
    """
    Si hay lat/lon -> rejilla; si no -> serie temporal promedio.
    species_name: "PM25" o "PM10" (solo para nombre y etiqueta)
    """
    sp = species_name.upper()
    if _pm_has_latlon(csv_path):
        df = _read_points(csv_path)
        if not {"lon","lat","value"}.issubset(df.columns):
            return {"ok": False, "msg": f"{sp}: faltan lon/lat/value"}
        pts = df[["lon","lat","value"]].dropna()
        if pts.empty:
            return {"ok": False, "msg": f"{sp}: sin puntos"}
        grid, extent, n = _grid_mean(pts, binsize_deg=0.05)
        out_png = Path(out_dir)/f"zone{zone_id}_{sp.lower()}_grid.png"
        png = _save_imshow(grid, extent, f"Zona {zone_id} • {sp} (rejilla 0.05°)", f"{sp} (µg/m³)", out_png, pclip=(2,98))
        return {"ok": True, "png": png, "n": n, "mean": float(np.nanmean(pts['value']))}
    else:
        out_png = Path(out_dir)/f"zone{zone_id}_{sp.lower()}_timeseries.png"
        png = _series_promedio_por_hora(csv_path, out_png, f"Zona {zone_id} • {sp} (promedio horario)", f"{sp} (µg/m³)")
        if png:
            # mean global de la ventana
            return {"ok": True, "png": png, "n": None, "mean": _mean_from_csv(csv_path)}
        return {"ok": False, "msg": f"{sp}: no se pudo graficar"}

# ---------- Dispatcher “una sola llamada” ----------
def plot_all_custom(zone_id, out_dir, *, res_all: dict, res_pm: dict | None):
    out = {}

    # NO2
    info = res_all.get("NO2", {})
    if info.get("ok") and info.get("csv"):
        out["NO2"] = plot_no2(zone_id, info["csv"], out_dir, binsize_deg=0.25, pclip=(2,98))
    else:
        out["NO2"] = {"ok": False, "msg": "NO2: sin csv"}

    # HCHO
    info = res_all.get("HCHO", {})
    if info.get("ok") and info.get("csv"):
        out["HCHO"] = plot_hcho(zone_id, info["csv"], out_dir, binsize_deg=0.25, pclip=(2,98))
    else:
        out["HCHO"] = {"ok": False, "msg": "HCHO: sin csv"}

    # AI
    info = res_all.get("AI", {})
    if info.get("ok") and info.get("csv"):
        out["AI"] = plot_ai(zone_id, info["csv"], out_dir, binsize_deg=0.25, pclip=(2,98))
    else:
        out["AI"] = {"ok": False, "msg": "AI: sin csv"}

    # O3
    info = res_all.get("O3", {})
    if info.get("ok") and info.get("csv"):
        out["O3"] = plot_o3(zone_id, info["csv"], out_dir, binsize_deg=0.2, pclip=(5,95))
    else:
        out["O3"] = {"ok": False, "msg": "O3: sin csv"}

    # PMs (si hubo OpenAQ)
    if isinstance(res_pm, dict) and res_pm.get("ok"):
        for spkey in ("PM25", "PM10"):
            i = res_pm.get(spkey, {})
            if i.get("ok") and i.get("csv"):
                out[spkey] = plot_pm(zone_id, i["csv"], out_dir, species_name=spkey)
            else:
                out[spkey] = {"ok": False, "msg": f"{spkey}: sin csv"}
    else:
        out["PM25"] = {"ok": False, "msg": "PM25: OpenAQ sin datos"}
        out["PM10"] = {"ok": False, "msg": "PM10: OpenAQ sin datos"}

    return out

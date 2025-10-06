
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import importlib
import sys
import json
import earthaccess as ea
import glob
import math

# server conectivity
from flask import Flask, app, request, jsonify
import zone_run as run_mod
from flask_cors import CORS

# MACROS
ZONE_ID = 0
END_UTC_STR = "2025-10-05 12:00:00"  
WINDOW_HOURS = 6
BASE_OUT = r"C:\Users\Admin\Downloads\Nasa\Resultados"
SECRETS_PATH = Path(r"C:\Users\Admin\Downloads\Nasa\tempo_secrets.json")

def load_earthdata_from_json(json_path: Path):
    """Lee usuario/clave del JSON y los exporta como variables de entorno."""
    if not json_path.is_file():
        return False, f"No existe el archivo de secretos: {json_path}"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Error leyendo JSON de secretos: {e}"

    user = data.get("earthdata_username")
    pwd  = data.get("earthdata_password")
    if not user or not pwd:
        return False, "Faltan 'earthdata_username' o 'earthdata_password' en el JSON."

    os.environ["EARTHDATA_USERNAME"] = user
    os.environ["EARTHDATA_PASSWORD"] = pwd
    return True, "OK"


def ensure_earthdata_env(json_path: Path):
    #print(f"[DEBUG] Leyendo secretos de: {json_path}  (exists={json_path.is_file()})")
    ok, msg = load_earthdata_from_json(json_path)
    if ok:
        print("[INFO] Credenciales Earthdata cargadas (estrategia=environment).")
        # Opcional: confirma que llegó el usuario (no imprimas la password)
        print(f"[DEBUG] EARTHDATA_USERNAME={os.environ.get('EARTHDATA_USERNAME')!r}")
    else:
        print("[WARN]", msg)


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/region-time', methods=['POST'])

def region_time():
    data = request.get_json(silent=True) or {}
    region = data.get("region")
    utc_time = data.get("utcTime")

    if region is None or utc_time is None:
        return jsonify({"ok": False, "reason": "Missing 'region' or 'utcTime'"}), 400

    try:
        zone_id = int(region)
    except Exception:
        return jsonify({"ok": False, "reason": "region must be an integer (zone id)"}), 400

    try:
        res = run_mod.run_all_for_zone(zone_id, utc_time, WINDOW_HOURS, BASE_OUT)
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ======== UTILS ========
def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def sanitize_for_dir(s: str) -> str:
    # reemplaza caracteres problemáticos en Windows
    bad = [(":", ""), ("/", "-"), ("\\", "-"), ("?", ""), ("*", ""), ("|", ""),
           ("<", ""), (">", ""), ('"', ''), (" ", "_")]
    out = s
    for a, b in bad:
        out = out.replace(a, b)
    return out

def compute_window(end_utc_str: str, hours: int):
    end_dt = datetime.strptime(end_utc_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(hours=hours)
    return start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# ======== MAIN ========
def main():
    # 1) Credenciales Earthdata
    ensure_earthdata_env(SECRETS_PATH)

    # 2) Carpetas de salida
    end_folder = sanitize_for_dir(END_UTC_STR)
    base_with_end = ensure_dir(os.path.join(BASE_OUT, end_folder))
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    out_dir = ensure_dir(os.path.join(base_with_end, f"{ts}_zone{ZONE_ID}"))

    # 3) Ventana (solo para log)
    start_isoz, end_isoz = compute_window(END_UTC_STR, WINDOW_HOURS)
    print(f"[INFO] ZONA={ZONE_ID}  Ventana UTC: {start_isoz} → {end_isoz}")
    print(f"[INFO] Carpeta salida: {out_dir}")

    # 4) TEMPO (NO2/HCHO/O3/AI)
    import zone_run as run_mod
    res_all = run_mod.run_all_for_zone(ZONE_ID, END_UTC_STR, WINDOW_HOURS, out_dir)

    # 5) OpenAQ (PM2.5/PM10 últimas 6h) + actualizar summary JSON con medias
    from pathlib import Path
    import json, pandas as pd
    try:
        from openaq_run import run_openaq_for_zone
        res_pm = run_openaq_for_zone(ZONE_ID, out_dir, hours=6, want_pm25=True, want_pm10=True)
    except Exception as e:
        print("[OpenAQ] Error o módulo ausente:", e)
        res_pm = None

    def _compute_mean_from_csv(csv_path: str, value_col: str = "value") -> float | None:
        try:
            df = pd.read_csv(csv_path)
            if df.empty or value_col not in df.columns:
                return None
            return float(pd.to_numeric(df[value_col], errors="coerce").mean(skipna=True))
        except Exception:
            return None

    def _update_summary_with_pm(summary_json_path: str, pm25_info: dict | None, pm10_info: dict | None):
        if not summary_json_path: return
        try:
            p = Path(summary_json_path)
            summary = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"species": {}}
        except Exception:
            summary = {"species": {}}

        def _entry(info):
            if not info or not info.get("ok"):
                return {"csv": None, "rows": 0, "mean": None}
            csvp = info.get("csv")
            meanv = _compute_mean_from_csv(csvp, "value") if csvp else None
            return {"csv": csvp, "rows": int(info.get("rows", 0)), "mean": meanv}

        summary.setdefault("species", {})
        summary["species"]["PM25"] = _entry(pm25_info)
        summary["species"]["PM10"] = _entry(pm10_info)

        Path(summary_json_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SUMMARY] PM añadidos en {summary_json_path}")

    if isinstance(res_pm, dict) and res_pm.get("ok"):
        _update_summary_with_pm(
            res_all.get("summary_json"),
            res_pm.get("PM25"),
            res_pm.get("PM10")
        )
    elif isinstance(res_pm, dict):
        print(f"[OpenAQ] {res_pm.get('reason','sin datos')}")
        for e in res_pm.get("errors", []):
            print("  ", e)

    # 6) GRAFICAR todo con un solo módulo
    from zone_graficar import plot_all_custom

    plots = plot_all_custom(
        ZONE_ID,
        out_dir,
        res_all=res_all,                            # dict que te devolvió zone_run.run_all_for_zone
        res_pm=(res_pm if isinstance(res_pm, dict) and res_pm.get("ok") else None)
    )

    for k, v in plots.items():
        if v.get("ok"):
            print(f"[PLOT {k}] → {v['png']} | mean={v.get('mean')}")
        else:
            print(f"[PLOT {k}] (skip) {v.get('msg','')}")


    # 7) Estado por especie (TEMPO)
    for sp in ("NO2", "HCHO", "O3", "AI"):
        info = res_all.get(sp, {})
        print(f"[{sp}] ok={info.get('ok')} rows={info.get('rows')} csv={info.get('csv')}")

    print("[DONE] main.")

def region_time():
    # Intentar leer JSON del body; si no viene, tomar query params
    data = request.get_json(silent=True) or {}
    region = data.get("region") or request.args.get("region")
    utcTime = data.get("utcTime") or request.args.get("utcTime")
    window = data.get("windowHours") or request.args.get("windowHours") or 6
    out_dir = data.get("outDir") or str(Path(BASE_OUT))  # opcional

    if region is None or utcTime is None:
        return jsonify({"ok": False, "reason": "Missing 'region' or 'utcTime'"}), 400

    try:
        zone_id = int(region)
    except Exception:
        return jsonify({"ok": False, "reason": "region must be an integer (zone id)"}), 400

    try:
        window = int(window)
    except Exception:
        window = 6

    try:
        # Llamada bloqueante a tu función; para tareas largas, ejecutar en background/job queue
        res = run_mod.run_all_for_zone(zone_id, utcTime, window, out_dir)
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



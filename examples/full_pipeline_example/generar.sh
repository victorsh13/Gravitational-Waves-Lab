#!/usr/bin/env bash
# Ejemplo completo, paso 1: valida la configuración y genera salidas/ejemplo.h5.
# Uso (desde cualquier carpeta, con el entorno de gwsim activado):
#   bash examples/full_pipeline_example/generar.sh
# Variables opcionales: PYTHON=/ruta/a/python (intérprete), WORKERS=N (procesos; por defecto 1).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"                                    # gwsim se ejecuta desde la raíz del repositorio
PY="${PYTHON:-python}"
OUT="$HERE/salidas/ejemplo.h5"

# 1. ¿El entorno tiene gwsim y sus dependencias?
if ! "$PY" -c "import gwsim, pycbc, h5py" 2>/dev/null; then
    echo "ERROR: '$PY' no puede importar gwsim/pycbc/h5py."
    echo "       Activa el entorno (conda activate gwsim, o source .venv/bin/activate) y repite."
    exit 1
fi

# 2. ¿Está el paquete de PSD? Si no, ruido analítico (curvas de diseño), con aviso.
CONFIG="$HERE/config.json"
if [ ! -f "$ROOT/psd_pack_o3b.npz" ]; then
    echo "AVISO: no encuentro $ROOT/psd_pack_o3b.npz (guía, paso 5)."
    echo "       Uso config_analitico.json: ruido con curvas de diseño, NO con las PSD de O3b."
    CONFIG="$HERE/config_analitico.json"
fi

# 3. gwsim nunca sobrescribe: si ya existe la salida, paramos y explicamos qué hacer.
if [ -e "$OUT" ]; then
    echo "ERROR: ya existe $OUT"
    echo "       Para generarlo de nuevo, bórralo antes:  rm \"$OUT\""
    exit 1
fi
mkdir -p "$HERE/salidas"

echo "== 1/3 Validando $(basename "$CONFIG")"
"$PY" -c "import sys; from gwsim.config import load_config; cfg, avisos = load_config(sys.argv[1]); print('Configuración válida:', cfg['n_samples'], 'señales, modo', cfg['mode'], ', salidas', cfg['outputs']); [print('AVISO:', a) for a in avisos]" "$CONFIG"

echo "== 2/3 Generando $OUT"
"$PY" -m gwsim generate --config "$CONFIG" --output "$OUT" --workers "${WORKERS:-1}"

echo "== 3/3 Resumen del fichero"
"$PY" -m gwsim info "$OUT"

echo
echo "Siguiente paso:  $PY ${HERE#"$ROOT"/}/leer_y_comprobar.py"

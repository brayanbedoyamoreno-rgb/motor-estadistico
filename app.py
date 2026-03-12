from flask import Flask, request, jsonify
import numpy as np
import scipy.stats as stats
import warnings
import ast
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ==========================================
# 1. FUNCIONES MATEMÁTICAS Y DE LIMPIEZA
# ==========================================
def convertir_a_minutos(tiempo_str):
    try:
        partes = str(tiempo_str).strip().replace("'", "").replace('"', '').split(':')
        if len(partes) == 3:
            return (int(partes[0]) * 60) + int(partes[1]) + (int(partes[2]) / 60.0)
        elif len(partes) == 2:
            return int(partes[0]) + (int(partes[1]) / 60.0)
        return 0.0
    except Exception: return 0.0

def pasa_aleatoriedad(datos, alpha=0.05):
    n = len(datos)
    if n < 4: return False
    mediana = np.median(datos)
    signos = [1 if x > mediana else (0 if x < mediana else None) for x in datos]
    signos = [s for s in signos if s is not None]
    if len(signos) < 2: return False
    rachas = 1
    for i in range(1, len(signos)):
        if signos[i] != signos[i-1]: rachas += 1
    n1, n2 = sum(signos), len(signos) - sum(signos)
    if n1 == 0 or n2 == 0: return False
    mean_runs = ((2 * n1 * n2) / (n1 + n2)) + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2)**2) * (n1 + n2 - 1))
    std_runs = np.sqrt(var_runs) if var_runs > 0 else 0
    z_mediana = (rachas - mean_runs) / std_runs if std_runs > 0 else 0
    p_mediana = 2 * (1 - stats.norm.cdf(abs(z_mediana)))
    
    turnings = sum(1 for i in range(1, n - 1) if (datos[i-1] < datos[i] > datos[i+1]) or (datos[i-1] > datos[i] < datos[i+1]))
    mean_turn = (2 * n - 1) / 3
    var_turn = (16 * n - 29) / 90
    std_turn = np.sqrt(var_turn) if var_turn > 0 else 0
    z_turn = (turnings - mean_turn) / std_turn if std_turn > 0 else 0
    p_turn = 2 * (1 - stats.norm.cdf(abs(z_turn)))
    return (p_mediana > alpha and p_turn > alpha)

def limpiar_outliers_iqr(datos):
    if len(datos) < 4: return datos
    q1, q3 = np.percentile(datos, [25, 75])
    iqr = q3 - q1
    return np.array([x for x in datos if (q1 - 1.5 * iqr) <= x <= (q3 + 1.5 * iqr)])

def agrupar_muestras_homogeneas(muestras_validas, alpha=0.05):
    nombres = list(muestras_validas.keys())
    similares = {nombre: set([nombre]) for nombre in nombres}
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            n1, n2 = nombres[i], nombres[j]
            _, p_val = stats.ks_2samp(muestras_validas[n1], muestras_validas[n2])
            if p_val > alpha:
                similares[n1].add(n2)
                similares[n2].add(n1)
    visitados, grupos = set(), []
    for nombre in nombres:
        if nombre not in visitados:
            g_actual, cola = set(), [nombre]
            while cola:
                act = cola.pop(0)
                if act not in visitados:
                    visitados.add(act); g_actual.add(act)
                    cola.extend(list(similares[act] - visitados))
            grupos.append(g_actual)
    pooled = {}
    for i, grupo in enumerate(grupos):
        n_grupo = " + ".join(sorted(list(grupo)))
        pooled[f"GRUPO {i+1} [{n_grupo}]"] = np.concatenate([muestras_validas[n] for n in grupo])
    return pooled

def ajustar_grupo(datos, nombre_grupo):
    datos_ord = np.sort(datos)
    N = len(datos_ord)
    distribuciones = {
        "Log-Logística": stats.fisk, "Lognormal": stats.lognorm, 
        "Weibull": stats.weibull_min, "Gamma": stats.gamma, "Normal": stats.norm
    }
    resultados = []
    for nombre, dist in distribuciones.items():
        try:
            params = dist.fit(datos, floc=0) if nombre != "Normal" else dist.fit(datos)
            ks_stat, _ = stats.kstest(datos, dist.name, args=params)
            resultados.append({"nombre": nombre, "dist_obj": dist, "params": params, "ks": ks_stat})
        except: continue
    resultados.sort(key=lambda x: x["ks"])
    mejor = resultados[0]
    return {
        "grupo_nombre": nombre_grupo,
        "cantidad_datos": N,
        "distribucion_ganadora": mejor['nombre'],
        "rango_60_minutos": [round(mejor['dist_obj'].ppf(0.20, *mejor['params']), 2), 
                             round(mejor['dist_obj'].ppf(0.80, *mejor['params']), 2)]
    }

# ==========================================
# 2. ENDPOINT PRINCIPAL
# ==========================================
@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        raw_data = request.json
        if not raw_data: return jsonify({"error": "No data"}), 400
        items = raw_data["data"] if isinstance(raw_data, dict) and "data" in raw_data else (raw_data if isinstance(raw_data, list) else [raw_data])

        dict_muestras = {}
        for fila in items:
            t_raw = next((v for k, v in fila.items() if any(x in k.upper() for x in ["TRAZA", "SLA"])), "")
            col_b = fila.get("ASIGNADO A OMT", "N/A")
            col_c = fila.get("CASUISTICA", "N/A")
            col_d = fila.get("OPS", "N/A")
            key = f"{col_b} | {col_c} | {col_d}"
            
            try:
                if isinstance(t_raw, list): lista = t_raw
                elif isinstance(t_raw, str) and t_raw.strip().startswith('['): lista = ast.literal_eval(t_raw)
                else: lista = [t_raw]
                minutos = [convertir_a_minutos(t) for t in lista if t]
                if minutos:
                    dict_muestras[key] = np.concatenate([dict_muestras.get(key, np.array([])), minutos])
            except: pass

        validas, rechazadas = {}, []
        for nombre, datos in dict_muestras.items():
            if len(datos) < 4:
                rechazadas.append({"nombre": nombre, "motivo": f"Datos insuficientes ({len(datos)})"})
                continue
            if pasa_aleatoriedad(datos):
                validas[nombre] = datos
            else:
                limpios = limpiar_outliers_iqr(datos)
                if len(limpios) >= 4 and pasa_aleatoriedad(limpios):
                    validas[nombre] = limpios
                else:
                    rechazadas.append({"nombre": nombre, "motivo": "Falla prueba de aleatoriedad (Datos sesgados)"})

        res_estadisticos = []
        if validas:
            pooled = agrupar_muestras_homogeneas(validas) if len(validas) > 1 else {list(validas.keys())[0]: list(validas.values())[0]}
            for n, d in pooled.items():
                res_estadisticos.append(ajustar_grupo(d, n))

        return jsonify({
            "status": "success",
            "resultados_estadisticos": res_estadisticos,
            "muestras_rechazadas": rechazadas
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "mensaje": str(e)}), 500

@app.route('/', methods=['GET'])
def health(): return jsonify({"status": "online"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))

from flask import Flask, request, jsonify
import numpy as np
import scipy.stats as stats
import warnings
import ast
import os
import json # Importante para reemplazar ast

# Desactivar advertencias matemáticas para mantener la consola limpia
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ==========================================
# 1. FUNCIONES MATEMÁTICAS Y DE LIMPIEZA
# (Mantenidas igual, ya están bien diseñadas)
# ==========================================

def convertir_a_minutos(tiempo_str):
    try:
        if not tiempo_str: return 0.0
        partes = str(tiempo_str).strip().replace("'", "").replace('"', '').split(':')
        if len(partes) == 3: # HH:MM:SS
            return (int(partes[0]) * 60) + int(partes[1]) + (int(partes[2]) / 60.0)
        elif len(partes) == 2: # MM:SS
            return int(partes[0]) + (int(partes[1]) / 60.0)
        return float(tiempo_str)
    except: return 0.0

def pasa_aleatoriedad(datos, alpha=0.05):
    n = len(datos)
    if n < 4: return False
    
    # Prueba de Rachas (Wald-Wolfowitz)
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
    
    # Prueba de Puntos de Giro (Tendencias)
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
    
    top_3 = [{"nombre": r["nombre"], "score": i+1} for i, r in enumerate(resultados[:3])]
    
    return {
        "grupo_nombre": nombre_grupo,
        "cantidad_datos": int(N),
        "distribucion_ganadora": mejor['nombre'],
        "rango_60_minutos": [round(float(mejor['dist_obj'].ppf(0.20, *mejor['params'])), 2), 
                             round(float(mejor['dist_obj'].ppf(0.80, *mejor['params'])), 2)],
        "top_3_distribuciones": top_3
    }

# ==========================================
# 2. ENDPOINT PRINCIPAL (API) OPTIMIZADO
# ==========================================

@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        raw_data = request.json
        if not raw_data: return jsonify({"error": "No data"}), 400
        
        items = raw_data.get("data", raw_data) if isinstance(raw_data, dict) else (raw_data if isinstance(raw_data, list) else [raw_data])

        # 1. OPTIMIZACIÓN: Pre-calcular los nombres de las columnas una sola vez
        if not items:
            return jsonify({"status": "success", "resultados_estadisticos": [], "muestras_rechazadas": []}), 200
            
        primera_fila = items[0].keys()
        col_t = next((k for k in primera_fila if any(x in str(k).upper() for x in ["TRAZA", "SLA", "TIEMPO"])), None)
        col_b = next((k for k in primera_fila if "ASIGNADO" in str(k).upper()), None)
        col_c = next((k for k in primera_fila if "CASUISTICA" in str(k).upper() or "CASO" in str(k).upper()), None)
        col_d = next((k for k in primera_fila if "OPS" in str(k).upper() or "OPERACION" in str(k).upper()), None)

        if not col_t:
            return jsonify({"error": "No se encontro columna de tiempo"}), 400

        # Usamos listas normales en Python (mucho más rápido que np.append en un bucle)
        dict_muestras_temp = {}

        for fila in items:
            # Extracción directa (O(1) en vez de O(N))
            t_raw = fila.get(col_t)
            v_b = fila.get(col_b, "Sin Asignado") if col_b else "Sin Asignado"
            v_c = fila.get(col_c, "Sin Caso") if col_c else "Sin Caso"
            v_d = fila.get(col_d, "N/A") if col_d else "N/A"
            
            key = f"{v_b} | {v_c} | {v_d}"
            if t_raw is None: continue

            procesados = []
            try:
                if isinstance(t_raw, list):
                    procesados = [convertir_a_minutos(t) for t in t_raw if t]
                elif isinstance(t_raw, str):
                    t_raw_str = t_raw.strip()
                    if t_raw_str.startswith('['):
                        # 2. OPTIMIZACIÓN: Usar json.loads es mucho más rápido que ast.literal_eval
                        lista_eval = json.loads(t_raw_str.replace("'", '"')) 
                        procesados = [convertir_a_minutos(t) for t in lista_eval if t]
                    else:
                        procesados = [convertir_a_minutos(t_raw_str)]
                else:
                    procesados = [convertir_a_minutos(t_raw)]
                
                # 3. OPTIMIZACIÓN: Usar .append de Python y convertir a Numpy al final
                if procesados:
                    if key not in dict_muestras_temp:
                        dict_muestras_temp[key] = []
                    dict_muestras_temp[key].extend(procesados)
            except Exception:
                pass

        # Convertir a Numpy arrays todo al final de una sola vez
        dict_muestras = {k: np.array(v) for k, v in dict_muestras_temp.items()}

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
                    rechazadas.append({"nombre": nombre, "motivo": "Falla prueba de aleatoriedad (Datos con sesgos/tendencias)"})

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

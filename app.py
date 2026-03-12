from flask import Flask, request, jsonify
import numpy as np
import scipy.stats as stats
import warnings
import ast
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)

# [TUS FUNCIONES MATEMÁTICAS ESTÁN AQUÍ OCULTAS PARA NO HACER LARGO EL CÓDIGO - EL SERVIDOR LAS CORRERÁ IGUAL]
def convertir_a_minutos(tiempo_str):
    try:
        partes = str(tiempo_str).strip().split(':')
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
    q1 = np.percentile(datos, 25)
    q3 = np.percentile(datos, 75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    return np.array([x for x in datos if limite_inf <= x <= limite_sup])

def agrupar_muestras_homogeneas(muestras_validas, alpha=0.05):
    nombres = list(muestras_validas.keys())
    similares = {nombre: set([nombre]) for nombre in nombres}
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            nombre1, nombre2 = nombres[i], nombres[j]
            _, p_val = stats.ks_2samp(muestras_validas[nombre1], muestras_validas[nombre2])
            if p_val > alpha:
                similares[nombre1].add(nombre2)
                similares[nombre2].add(nombre1)
    visitados = set()
    grupos = []
    for nombre in nombres:
        if nombre not in visitados:
            grupo_actual = set()
            cola = [nombre]
            while cola:
                actual = cola.pop(0)
                if actual not in visitados:
                    visitados.add(actual)
                    grupo_actual.add(actual)
                    cola.extend(list(similares[actual] - visitados))
            grupos.append(grupo_actual)
    datos_pooled = {}
    for i, grupo in enumerate(grupos):
        nombres_grupo = " + ".join(sorted(list(grupo)))
        datos_combinados = np.concatenate([muestras_validas[n] for n in grupo])
        datos_pooled[f"GRUPO {i+1} [{nombres_grupo}]"] = datos_combinados
    return datos_pooled

def ajustar_grupo(datos, nombre_grupo):
    datos_ordenados = np.sort(datos)
    N = len(datos_ordenados)
    distribuciones = {
        "Weibull Inversa": stats.invweibull, "Log-Logística": stats.fisk,
        "Lognormal": stats.lognorm, "Gaussiana Inversa": stats.invgauss,
        "Weibull": stats.weibull_min, "Exponencial": stats.expon,
        "Gamma": stats.gamma, "Normal": stats.norm
    }
    resultados = []
    for nombre, dist in distribuciones.items():
        try:
            params = dist.fit(datos, floc=0) if nombre != "Normal" else dist.fit(datos)
            ks_stat, _ = stats.kstest(datos, dist.name, args=params)
            cdf_vals = np.clip(dist.cdf(datos_ordenados, *params), 1e-10, 1 - 1e-10)
            i = np.arange(1, N + 1)
            S = np.sum(((2 * i - 1) / N) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1])))
            ad_stat = -N - S
            resultados.append({"nombre": nombre, "dist_obj": dist, "params": params, "ks": ks_stat, "ad": ad_stat})
        except: continue
    if not resultados: return {"grupo_nombre": nombre_grupo, "error": "Fallo ajuste"}
    resultados.sort(key=lambda x: x["ks"])
    for idx, r in enumerate(resultados): r["rank_ks"] = idx + 1
    resultados.sort(key=lambda x: x["ad"])
    for idx, r in enumerate(resultados): r["rank_ad"] = idx + 1
    for r in resultados: r["score"] = (r["rank_ks"] * 0.4) + (r["rank_ad"] * 0.6)
    resultados.sort(key=lambda x: x["score"])
    mejor = resultados[0]
    min_60 = mejor['dist_obj'].ppf(0.20, *mejor['params'])
    max_60 = mejor['dist_obj'].ppf(0.80, *mejor['params'])
    return {
        "grupo_nombre": nombre_grupo,
        "cantidad_datos": N,
        "distribucion_ganadora": mejor['nombre'],
        "rango_60_minutos": [round(min_60, 2), round(max_60, 2)],
        "top_3_distribuciones": [{"nombre": r["nombre"], "score": round(r["score"], 2)} for r in resultados[:3]]
    }

# ==========================================
# RUTA API EN MODO "RAYOS X"
# ==========================================
@app.route('/analizar', methods=['POST'])
def analizar_datos():
    try:
        # DIAGNÓSTICO 1: ¿Qué llegó realmente?
        datos_recibidos = request.data.decode('utf-8')
        if not request.is_json:
            return jsonify({
                "DEBUG_ERROR": "La petición NO llegó como JSON",
                "contenido_recibido": datos_recibidos[:500]
            }), 400

        datos_n8n = request.json
        
        if not datos_n8n:
            return jsonify({"DEBUG_ERROR": "El JSON llegó completamente vacío (null)"}), 400

        if isinstance(datos_n8n, dict) and "data" in datos_n8n:
            datos_n8n = datos_n8n["data"]
        elif isinstance(datos_n8n, dict):
            datos_n8n = [datos_n8n]

        # DIAGNÓSTICO 2: Ver un ejemplo de lo que se leyó
        ejemplo_fila = datos_n8n[0] if len(datos_n8n) > 0 else "Lista vacía"

        todas_las_muestras = {}
        errores_de_lectura = []
        
        for fila in datos_n8n:
            texto_tiempos = col_b = col_c = col_d = ""
            for key, val in fila.items():
                k_upper = str(key).upper()
                if "TRAZA" in k_upper or "SLA" in k_upper: texto_tiempos = val
                elif "ASIGNADO" in k_upper: col_b = val
                elif "CASUISTICA" in k_upper: col_c = val
                elif "OPS" in k_upper: col_d = val
            
            if not texto_tiempos:
                continue

            nombre_muestra = f"{col_b} | {col_c} | {col_d}"
            
            try:
                lista_tiempos_str = ast.literal_eval(str(texto_tiempos))
                tiempos_minutos = [convertir_a_minutos(t) for t in lista_tiempos_str]
                if len(tiempos_minutos) > 0:
                    if nombre_muestra in todas_las_muestras:
                        todas_las_muestras[nombre_muestra] = np.concatenate([todas_las_muestras[nombre_muestra], np.array(tiempos_minutos)])
                    else:
                        todas_las_muestras[nombre_muestra] = np.array(tiempos_minutos)
            except Exception as e:
                errores_de_lectura.append(str(e))

        # DIAGNÓSTICO 3: Si no se procesó nada, decir el porqué
        if len(todas_las_muestras) == 0:
            return jsonify({
                "DEBUG_ERROR": "Se leyó el JSON, pero ninguna fila tenía el formato correcto.",
                "ejemplo_fila_1": ejemplo_fila,
                "errores_ast_literal_eval": errores_de_lectura[:3]
            }), 200

        # Fases matemáticas normales
        muestras_validas = {}
        for nombre, datos in todas_las_muestras.items():
            if pasa_aleatoriedad(datos): muestras_validas[nombre] = datos
            else:
                datos_limpios = limpiar_outliers_iqr(datos)
                if len(datos_limpios) >= 4 and pasa_aleatoriedad(datos_limpios):
                    muestras_validas[nombre] = datos_limpios

        resultados_finales = []
        if muestras_validas:
            if len(muestras_validas) == 1:
                datos_pooled = {list(muestras_validas.keys())[0]: list(muestras_validas.values())[0]}
            else:
                datos_pooled = agrupar_muestras_homogeneas(muestras_validas)

            for nombre_grupo, datos_combinados in datos_pooled.items():
                res_grupo = ajustar_grupo(datos_combinados, nombre_grupo)
                resultados_finales.append(res_grupo)

        return jsonify({
            "status": "success",
            "muestras_totales_leidas": len(todas_las_muestras),
            "resultados_estadisticos": resultados_finales
        }), 200

    except Exception as e:
        import traceback
        return jsonify({
            "DEBUG_ERROR": "Excepción general en el servidor",
            "detalle": str(e),
            "traza": traceback.format_exc()
        }), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

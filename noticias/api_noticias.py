# api_noticias.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib

app = FastAPI(title="Detector de noticias falsas")
modelo = joblib.load("modelo_noticias.joblib")


class Noticia(BaseModel):
    texto: str = Field(..., min_length=20, max_length=10000)


@app.post("/analizar")
def analizar(noticia: Noticia):
    prob_falsa = float(modelo.predict_proba([noticia.texto])[0][1])
    return {
        "veredicto": "FALSA" if prob_falsa >= 0.5 else "REAL",
        "probabilidad_falsa": round(prob_falsa, 4),
    }


PAGINA = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Detector de noticias falsas</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Segoe UI", Arial, sans-serif;
    background: #eef2f7;
    color: #1f2937;
    min-height: 100vh;
  }
  header {
    background: #1e3a8a;
    color: white;
    padding: 20px 24px;
  }
  header h1 { margin: 0; font-size: 22px; }
  header p { margin: 4px 0 0; opacity: 0.85; font-size: 14px; }
  main {
    max-width: 780px;
    margin: 28px auto;
    padding: 0 16px;
  }
  .tarjeta {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 20px;
  }
  label { font-weight: 600; display: block; margin-bottom: 8px; }
  textarea {
    width: 100%;
    min-height: 180px;
    padding: 12px;
    font-size: 15px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    resize: vertical;
    font-family: inherit;
  }
  textarea:focus { outline: 2px solid #3b82f6; border-color: transparent; }
  .contador { font-size: 13px; color: #64748b; margin-top: 6px; }
  button {
    margin-top: 14px;
    background: #1e3a8a;
    color: white;
    border: none;
    padding: 12px 22px;
    font-size: 16px;
    border-radius: 8px;
    cursor: pointer;
  }
  button:hover { background: #1e40af; }
  button:disabled { background: #94a3b8; cursor: not-allowed; }
  #resultado { display: none; }
  .veredicto {
    font-size: 26px;
    font-weight: 700;
    padding: 14px 18px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 14px;
  }
  .real { background: #dcfce7; color: #166534; }
  .falsa { background: #fee2e2; color: #991b1b; }
  .barra { background: #e2e8f0; border-radius: 8px; height: 18px; overflow: hidden; }
  .barra div { height: 100%; background: #dc2626; transition: width 0.5s; }
  .detalle { font-size: 14px; color: #475569; margin-top: 8px; }
  .error { color: #b91c1c; display: none; margin-top: 10px; }
  footer { text-align: center; font-size: 13px; color: #64748b; padding: 20px; }
</style>
</head>
<body>
<header>
  <h1>Detector de noticias falsas</h1>
  <p>Pega el texto de una noticia y te diremos si parece real o falsa.</p>
</header>

<main>
  <div class="tarjeta">
    <label for="texto">Texto de la noticia</label>
    <textarea id="texto" placeholder="Pega aquí el texto completo de la noticia..." oninput="actualizarContador()"></textarea>
    <div class="contador" id="contador">0 caracteres (mínimo 20)</div>
    <button id="boton" onclick="analizar()">Analizar noticia</button>
    <div class="error" id="error"></div>
  </div>

  <div class="tarjeta" id="resultado">
    <div class="veredicto" id="veredicto"></div>
    <div>Probabilidad de ser falsa: <strong id="porcentaje"></strong></div>
    <div class="barra"><div id="barra" style="width: 0%"></div></div>
    <div class="detalle">Resultado del modelo. Es una estimación, no una verificación de hechos.</div>
  </div>
</main>

<footer>Modelo de prueba. Para resultados confiables necesita entrenarse con más noticias.</footer>

<script>
function actualizarContador() {
  const n = document.getElementById('texto').value.trim().length;
  document.getElementById('contador').textContent = n + ' caracteres (mínimo 20)';
}

function mostrarError(msg) {
  const e = document.getElementById('error');
  e.textContent = msg;
  e.style.display = 'block';
}

async function analizar() {
  const texto = document.getElementById('texto').value.trim();
  document.getElementById('error').style.display = 'none';
  document.getElementById('resultado').style.display = 'none';

  if (texto.length < 20) {
    mostrarError('Escribe al menos 20 caracteres.');
    return;
  }

  const boton = document.getElementById('boton');
  boton.disabled = true;
  boton.textContent = 'Analizando...';

  try {
    const r = await fetch('/analizar', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texto: texto })
    });
    if (!r.ok) throw new Error('El servidor respondió con error ' + r.status);
    const datos = await r.json();

    const v = document.getElementById('veredicto');
    v.textContent = datos.veredicto === 'FALSA' ? 'Posiblemente FALSA' : 'Parece REAL';
    v.className = 'veredicto ' + (datos.veredicto === 'FALSA' ? 'falsa' : 'real');

    const pct = (datos.probabilidad_falsa * 100).toFixed(1);
    document.getElementById('porcentaje').textContent = pct + '%';
    document.getElementById('barra').style.width = pct + '%';
    document.getElementById('resultado').style.display = 'block';
  } catch (err) {
    mostrarError('No se pudo analizar: ' + err.message);
  } finally {
    boton.disabled = false;
    boton.textContent = 'Analizar noticia';
  }
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def inicio():
    return PAGINA
import json
import os
import tensorflow as tf
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.core.config import model_registry
from app.routers import whatif, classify, forecast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# HARDCORE PATCH: Pencegah Bug Keras 3
# =========================================================
original_dense_init = tf.keras.layers.Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
tf.keras.layers.Dense.__init__ = patched_dense_init
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[BOOTING] Memuat MODEL AI FinTime ke memori server")
    try:
        # --- LOAD MODEL 1 ---
        nlp_keras = os.path.join(BASE_DIR, "models", "classify", "model_classify.keras")
        nlp_json = os.path.join(BASE_DIR, "models", "classify", "metadata.json")
        model_registry.classify_model = tf.keras.models.load_model(nlp_keras, compile=False)
        with open(nlp_json, "r") as f:
            model_registry.classify_metadata = json.load(f)
        print("[READY] Model 1 (NLP Classification) Berhasil Dimuat.")

        # --- LOAD MODEL 2 ---
        nlp_keras = os.path.join(BASE_DIR, "models", "forecast", "model_forecast.keras")
        nlp_json = os.path.join(BASE_DIR, "models", "forecast", "metadata.json")
        model_registry.forecast_model = tf.keras.models.load_model(nlp_keras, compile=False)
        with open(nlp_json, "r") as f:
            model_registry.forecast_metadata = json.load(f)
        print("[READY] Model 2 (NLP Forecasting) Berhasil Dimuat.")

                # --- LOAD MODEL 3 ---
        nlp_keras = os.path.join(BASE_DIR, "models", "whatif", "model_whatif.keras")
        nlp_json = os.path.join(BASE_DIR, "models", "whatif", "metadata.json")
        model_registry.whatif_model = tf.keras.models.load_model(nlp_keras, compile=False)
        with open(nlp_json, "r") as f:
            model_registry.whatif_metadata = json.load(f)
        print("[READY] Model 3 (NLP What-If) Berhasil Dimuat.")
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Gagal memuat biner AI Monolith: {e}")
    
    yield
    print("[SHUTDOWN] Membersihkan resource memori server")

app = FastAPI(title="FinTime Dedicated AI Engine", lifespan=lifespan)

# Daftarkan Seluruh Router Resmi
app.include_router(classify.router)
app.include_router(whatif.router)
app.include_router(forecast.router)


@app.get("/health")
def health():
    return {"status": "healthy", "engine": "FastAPI Monolith Inference Server Ready"}
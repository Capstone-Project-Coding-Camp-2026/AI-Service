import numpy as np
import re
from fastapi import HTTPException
from app.core.config import model_registry
from app.schemas.classify import ClassifyRequest

def process_classification(req: ClassifyRequest):
    if model_registry.classify_model is None or model_registry.classify_metadata is None:
        raise HTTPException(status_code=500, detail="Model Classify belum siap.")
    
    try:
        vocab = model_registry.classify_metadata.get("vocab")
        expected_dim = model_registry.classify_model.input_shape[-1] 
        
        input_vector = np.zeros((1, expected_dim), dtype=np.float32)
        clean_text = req.desc.lower()
        words = re.findall(r'\b\w+\b', clean_text)
        
        for word in words:
            if word in vocab:
                idx = vocab[word]
                if idx < expected_dim:
                    input_vector[0, idx] = 1.0
                
        predictions = model_registry.classify_model.predict(input_vector, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence_score = float(np.max(predictions[0]))
        
        class_names = model_registry.classify_metadata.get("class_names", [])
        predicted_category = class_names[predicted_index] if predicted_index < len(class_names) else f"Category_{predicted_index}"

        return {
            "success": True,
            "description": req.desc,
            "predicted_category": predicted_category,
            "confidence": confidence_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal prediksi NLP: {str(e)}")
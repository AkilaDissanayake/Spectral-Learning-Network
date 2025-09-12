from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model("modulation_model.keras")
Selected_mods=['QPSK', 'PAM4', 'AM-DSB', 'GFSK', 'QAM64', 'AM-SSB', '8PSK', 'QAM16', 'WBFM', 'CPFSK', 'BPSK']
# Define a Pydantic model for request body
class PredictRequest(BaseModel):
    data: list

@app.post("/predict")

def predict_modulation(request: PredictRequest):
    # First rows are IQ data   
    sample = np.array(request.data[:-1],dtype=np.float32)

    # Last row is SNR
    SNR = request.data[-1][0]
    
    

    size = sample.shape
    if size != (2, 128):
        if size == (128, 2):
            sample = sample.T
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input shape {size}, expected (2,128) or (128,2)"
            )

     
    sample = np.expand_dims(sample, axis=0)  # (1,2,128)
    snr_input = np.array([[SNR]], dtype=np.float32)  # (1,1)

    # Predict
    pred = model.predict({"IQ_Input": sample, "SNR_Input": snr_input})
    modulation_class = int(np.argmax(pred))
    confidence = float(np.max(pred))
    return {"modulation": Selected_mods[modulation_class], "confidence": confidence}

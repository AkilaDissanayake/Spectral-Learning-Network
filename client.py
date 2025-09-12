import requests
import numpy as np

url = "http://127.0.0.1:8000/predict"

# Generate IQ sample (2x128)
iq_sample = np.random.rand(2, 128).tolist()

# SNR value
snr_value = [5]

# Combine IQ sample and SNR
data = iq_sample + [snr_value]

# Send POST request
response = requests.post(url, json={"data": data})

print(response.status_code)
print(response.json())
#uvicorn main:app --reload

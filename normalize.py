import glob
from sklearn.metrics import classification_report
import pandas as pd
import time
def normalize(path):
    try:
        img = Image.open(path).convert('RGB').resize((256, 256), resample=Image.Resampling.BICUBIC)
        input_data = np.asarray(img, dtype=np.uint8)
        '''
        if input_dtype == np.uint8 or input_dtype == np.int8:
            input_data = input_data.astype(input_dtype)
        else:
            if np.max(input_data) > 1.0:
                input_data /= 255.0
                input_data = input_data.astype(np.float32)'
        '''
        input_data = np.expand_dims(input_data, axis=0)
        return input_data
    except Exception:
        return None
    

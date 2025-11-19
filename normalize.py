import glob
from sklearn.metrics import classification_report
import pandas as pd
import time
import tensorflow as tf
import numpy as np

DIR = "/content/drive/MyDrive/YOLO-Plant-Pathology/PP_dataset/"
CLASS_NAMES = os.listdir(os.path.join(DIR, 'test'))
CLASS_NAMES.sort()
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']

def predict_tflite(image_path):
    img = Image.open(image_path).convert('RGB').resize((256, 256), resample=Image.Resampling.BICUBIC)
    input_data = np.array(img, dtype=np.float32)
    if input_dtype == np.uint8 or input_dtype == np.int8:
        input_data = input_data.astype(input_dtype)
    else:
        if np.max(input_data) > 1.0:
            input_data /= 255.0
        input_data = input_data.astype(np.float32)
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    probabilities = tf.nn.softmax(output_data[0])
    predicted_index = np.argmax(probabilities)
    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, probabilities[predicted_index]
test_img = os.path.join(DIR, 'test', CLASS_NAMES[1], '9e0ba1619bdf4943.jpg')
print(test_img)

test2 = "/content/drive/MyDrive/YOLO-Plant-Pathology/TEST_ADDITION/test.jpg"
print(f"Test image predicted as: {predict_tflite(test2)[0]}")
    

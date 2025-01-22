import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Load sẵn các mô hình
MODELS = {
    "DenseNet": load_model('models/densenet_model.h5'),
    "Resnet50": load_model('models/resnet50_model.h5'),
    "Vgg19": load_model('models/vgg19_model.h5'),
}

def predict_image(image_path, model_name):
    if model_name not in MODELS:
        return {"error": "Chưa chọn mô hình máy học nào! Hãy chọn 1 mô hình để dự đoán."}
    model = MODELS[model_name]
    
    # Tiền xử lý ảnh
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán
    predictions = model.predict(img_array)
    
    class_names = ['NORMAL', 'COVID-19', 'PNEUMONIA (VIÊM PHỔI)']
    
    # Trả về kết quả dự đoán dưới dạng lớp và xác suất
    class_probabilities = predictions.tolist()[0]  
    
    
    result = {class_names[i]: round(class_probabilities[i], 4) for i in range(len(class_names))}
    
    return result


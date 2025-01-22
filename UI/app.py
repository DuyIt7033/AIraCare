import os
from PIL import Image
from utils.prediction import predict_image
from flask import Flask, render_template, send_from_directory, flash, redirect, url_for, request
import zipfile
from werkzeug.utils import secure_filename
from skimage.feature import *
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    model_name = None
    image_name = None  
    max_class = None  # Lớp dự đoán cao nhất
    if request.method == "POST":
        image = request.files.get('image')

        if not handle_image_upload(image):
            return render_template('index.html', prediction=prediction, image_name=image_name)

        # Lấy mô hình được chọn
        model_name = request.form["model"]

        image_path = save_image(image)
        image_name = image.filename  

        if not is_xray_image(image_path):
            flash('Ảnh tải lên không phải là ảnh x-quang', 'error')
            return render_template('index.html', prediction=prediction, image_name=image_name)

        prediction = predict_image(image_path, model_name=model_name)

         # Xác định lớp có xác suất cao nhất
        if prediction:
            max_class = max(prediction, key=prediction.get)
    return render_template("index.html", prediction=prediction, model_name=model_name, image_name=image_name, max_class=max_class)

def handle_image_upload(image):
    """Kiểm tra ảnh đã được chọn và có hợp lệ không"""
    if not image:
        flash('Chưa chọn ảnh !', 'error')
        return False
    
    if image.filename == '':
        flash('Chưa chọn ảnh !', 'error')
        return False
    
    if not allowed_file(image.filename):
        flash('Chỉ nhận được file jpg, png, jpeg', 'error')
        return False
    
    return True


def allowed_file(filename):
    """Kiểm tra file có phải là ảnh hoặc ZIP"""
    ALLOWED_EXTENSIONS_IMAGES = {'png', 'jpg', 'jpeg'}
    ALLOWED_EXTENSIONS_ARCHIVES = {'zip'}

    # Kiểm tra nếu tệp là ảnh hoặc tệp ZIP
    if '.' in filename:
        extension = filename.rsplit('.', 1)[1].lower()
        return extension in ALLOWED_EXTENSIONS_IMAGES or extension in ALLOWED_EXTENSIONS_ARCHIVES
    return False

def save_image(image):
    """Lưu ảnh vào thư mục uploads và trả về đường dẫn"""
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)
    return image_path

def is_xray_image(image_path):
    """
    Kiểm tra xem ảnh có phải là ảnh X-quang 
    1. Độ tương phản.
    2. Tỷ lệ khung hình.
    3. Mật độ cạnh (Canny Edge).
    """
    try:
        # Đọc ảnh
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        
        # Độ tương phản
        contrast = img.max() - img.min()
        if contrast < 150: 
            return False

        # Tỷ lệ khung hình
        height, width = img.shape
        aspect_ratio = width / height
        if aspect_ratio < 0.8 or aspect_ratio > 1.2: 
            return False

        # Mật độ cạnh (Canny Edge)
        edges = cv2.Canny(img, threshold1=50, threshold2=150)
        edge_density = np.sum(edges) / (height * width)
        if edge_density < 3 or edge_density > 50:  
            return False


    except Exception as e:
        print(f"Lỗi khi kiểm tra ảnh X-quang: {e}")
        return False

    return True

@app.route("/upload-zip", methods=["GET", "POST"])
def upload_zip():
    model_name = None
    results = None

    if request.method == "POST":
        model_name = request.form.get("model")
        zip_file = request.files.get("zip_file")

        if not zip_file or zip_file.filename == '':
            flash("Chưa chọn tệp !", "error")
            return redirect(url_for('upload_zip'))

        if not allowed_file(zip_file.filename):
            flash("Tệp tải lên không phải zip!", "error")
            return redirect(url_for('upload_zip'))

        # Tạo thư mục để lưu tệp và giải nén
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(zip_file.filename))
        zip_file.save(zip_path)

        try:
             # Giải nén tệp
            if zip_path.endswith('.zip'):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(app.config['UPLOAD_FOLDER'])

            # Duyệt qua các ảnh trong thư mục giải nén và dự đoán
            image_results = {}
            for root, _, files in os.walk(app.config['UPLOAD_FOLDER']):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, file)
                        prediction = predict_image(image_path, model_name=model_name)

                        # Xác định lớp dự đoán cao nhất
                        if prediction:
                            max_class = max(prediction, key=prediction.get)
                            if max_class in image_results:
                                image_results[max_class] += 1
                            else:
                                image_results[max_class] = 1

            # Giải phóng bộ nhớ
            os.remove(zip_path)
            for root, _, files in os.walk(app.config['UPLOAD_FOLDER']):
                for file in files:
                    os.remove(os.path.join(root, file))

            results = image_results

        except Exception as e:
            flash(f"Lỗi khi xử lý tệp : {e}", "error")
            return redirect(url_for('upload_zip'))

    return render_template("upload_zip.html", model_name=model_name, results=results)

if __name__ == "__main__":
    app.run(debug=True)

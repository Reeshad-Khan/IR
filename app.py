import os
import shutil
import cv2

from skimage.metrics import structural_similarity as ssim
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import os
from cooccurrence.cooccurrences import ini_cooc_filter, calc_spatial_cooc
import os
app = Flask(__name__)

@app.context_processor
def inject_os():
    return dict(os=os)

UPLOAD_FOLDER = 'uploads'
REFERENCE_FOLDER = '/home/rk010/Desktop/test/data/data'
MOST_SIMILAR_FOLDER = 'most_similar'  # New folder for most similar images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MOST_SIMILAR_FOLDER'] = MOST_SIMILAR_FOLDER

COOC_FILTER_PATH = "./data/weights_cooc_44_best_model_8192_ft.npy"
VGG16 = "http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth"

# Define VGG16_net and cooc_filter globally
VGG16_net = models.vgg16(pretrained=False).features
model_dir = os.path.join(os.path.join(os.getcwd(), 'data'), 'networks')
VGG16_net.load_state_dict(model_zoo.load_url(VGG16, model_dir=model_dir))
VGG16_net.cuda()
VGG16_net.eval()

cooc_filter = np.load(COOC_FILTER_PATH)
cooc_filter = torch.FloatTensor(cooc_filter).cuda()

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def remove_files(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)




def calculate_similarity(cooc_map_uploaded, cooc_map_reference):
    # Ensure cooc_map_uploaded and cooc_map_reference have the same shape
    if cooc_map_uploaded.shape != cooc_map_reference.shape:
        cooc_map_uploaded = cooc_map_uploaded[:cooc_map_reference.shape[0], :cooc_map_reference.shape[1]]

    # Resize images to a common size (e.g., (256, 256))
    common_size = (256, 256)
    cooc_map_uploaded_resized = cv2.resize(cooc_map_uploaded, common_size)
    cooc_map_reference_resized = cv2.resize(cooc_map_reference, common_size)

    # Calculate SSIM
    similarity = ssim(cooc_map_uploaded_resized, cooc_map_reference_resized, multichannel=True)

    return similarity


def find_most_similar_image(cooc_map_uploaded):
    most_similar_image = None
    min_similarity = float('inf')

    for filename in list_files(REFERENCE_FOLDER):
        reference_image_path = os.path.join(REFERENCE_FOLDER, filename)

        # Load reference image and calculate co-occurrence maps
        reference_image = Image.open(reference_image_path)
        reference_image = np.array(reference_image)
        net_reference_image = np.moveaxis(np.array(reference_image), 2, 0)
        net_input_reference = torch.FloatTensor(np.expand_dims(net_reference_image, axis=0)).cuda()
        act_map_reference = VGG16_net(net_input_reference)
        cooc_map_reference = calc_spatial_cooc(act_map_reference, cooc_filter, 4)
        cooc_map_reference = cooc_map_reference.cpu().data.squeeze().numpy()

        # Calculate similarity
        similarity = calculate_similarity(cooc_map_uploaded, cooc_map_reference)

        # Update most similar image if needed
        if similarity < min_similarity:
            min_similarity = similarity
            most_similar_image = filename

    return most_similar_image

def save_most_similar_image(image_path, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Copy the image to the destination folder
    destination_path = os.path.join(destination_folder, os.path.basename(image_path))
    shutil.copy(image_path, destination_path)

    return destination_path


@app.route('/')
def index():
    files = list_files(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', uploaded_image=None, files=files, most_similar_image=None, os=os)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    # Remove existing files before saving the new one
    remove_files(app.config['UPLOAD_FOLDER'])
    remove_files(app.config['MOST_SIMILAR_FOLDER'])  # Remove existing most similar images

    if file:
        # List existing files before saving the new one
        existing_files = list_files(app.config['UPLOAD_FOLDER'])
        print("Existing Files:", existing_files)

        # Save the new file
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Load uploaded image and calculate co-occurrence maps
        uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_image = Image.open(uploaded_image_path)
        uploaded_image = np.array(uploaded_image)
        net_uploaded_image = np.moveaxis(np.array(uploaded_image), 2, 0)
        net_input_uploaded = torch.FloatTensor(np.expand_dims(net_uploaded_image, axis=0)).cuda()
        act_map_uploaded = VGG16_net(net_input_uploaded)
        cooc_map_uploaded = calc_spatial_cooc(act_map_uploaded, cooc_filter, 4)
        cooc_map_uploaded = cooc_map_uploaded.cpu().data.squeeze().numpy()

        # Find the most similar image in the reference folder
        most_similar_image = find_most_similar_image(cooc_map_uploaded)

        # Save the most similar image to a new folder
        most_similar_path = save_most_similar_image(
            os.path.join(REFERENCE_FOLDER, most_similar_image),
            app.config['MOST_SIMILAR_FOLDER']
        )

        # List files after saving the new one
        new_existing_files = list_files(app.config['UPLOAD_FOLDER'])
        print("New Existing Files:", new_existing_files)

        return render_template('index.html', uploaded_image=filename, files=new_existing_files, most_similar_image=most_similar_path)

@app.route('/uploads/<filename>')
def serve_uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/most_similar/<filename>')
def serve_most_similar_image(filename):
    return send_from_directory(app.config['MOST_SIMILAR_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

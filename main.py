import cv2
import os
import secrets

from flask import Flask, render_template, Response, send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from utils.face_detection import FaceDetection


# Generate a secure random key
secret_key = secrets.token_hex(16)

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key
app.config['UPLOADED_PHOTOS_DEST'] = './static/uploads'
app.config['RESULTS_DEST'] = './static/results'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)


class UploadForm(FlaskForm):
    """
    Represents a form for uploading a photo.

    Attributes:
        photo (FileField): The field for selecting a photo file.
        submit (SubmitField): The button for submitting the form.
    """
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Image only!'),
            FileRequired('File was empty!')
        ]
    )
    submit = SubmitField('Upload')

@app.route('/upload/<filename>')
def get_upload(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route('/results/<result_name>')
def get_result(result_name):
    return send_from_directory(app.config['RESULTS_DEST'], result_name)


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    """
    Uploads an image and performs face detection on it.

    Returns:
        Response: The annotated image with detected faces as a JPEG response.
    """
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        file_url = url_for('get_upload', filename=filename)
        # Make instance of FaceDetection class
        face_detection = FaceDetection(file_path)
        # Detect face
        result = face_detection.detect_face()
        # Visualize the result
        annotated_image = result.plot()
        # Save the annotated image
        result_name = filename.split('.')[0] + '_anotated.jpg'
        cv2.imwrite(app.config['RESULTS_DEST']+'/'+result_name, annotated_image)
        result_url = url_for('get_result', result_name=result_name)
        # # Encode the image as JPEG
        # _, buffer = cv2.imencode('.jpg', annotated_image)
        # # Convert the buffer to bytes
        # response_data = buffer.tobytes()
        # # Send the image as a response
        # return Response(response_data, content_type='image/jpeg')
    else:
        file_url = None
        result_url = None
    return render_template('index.html', form=form, file_url=file_url ,result_url=result_url)


if __name__ == '__main__':
    app.run(debug=True)

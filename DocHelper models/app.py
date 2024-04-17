from flask import Flask, render_template, request, jsonify
from bert_model import find_answer_pdf, initialize_model
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

model_path = os.getcwd()
upload_folder = os.path.join(os.getcwd(), 'Uploads')
app.config['UPLOAD_FOLDER'] = upload_folder
model, tokenizer = initialize_model(model_path=model_path)

if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/qa', methods=['POST'])
def qa():
    # Get data from the request
    question = request.form['question']

    # Checking for errors
    if 'pdf_file' not in request.files:
        return jsonify({'answer': 'No PDF file provided'})
    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return jsonify({'answer': 'No PDF file provided'})

    # Saving the PDF
    filename = secure_filename(pdf_file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf_file.save(pdf_path)

    # Calling the find_answer_pdf function
    answer = find_answer_pdf(question, pdf_path, model, tokenizer)

    # Return the answer as JSON
    return jsonify({'answer': answer})


if __name__ == '__main__':
    app.run(debug=True)

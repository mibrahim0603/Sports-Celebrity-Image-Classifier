from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import util
import os

app = Flask(__name__)
CORS(app)  # Globally opens up communication channels between frontend port 5500 & backend port 5000


@app.route('/classify_image', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
        # Safely capture the Base64 image payload from the jQuery POST request
        image_data = request.form.get('image_data')
        if not image_data:
            return jsonify({"error": "Missing image_data payload"}), 400

        response_data = util.classify_image(image_data)
        return jsonify(response_data)
    return jsonify({"message": "Send a POST request containing 'image_data' string base64 payload."})


if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    # Runs locally on port 5000
    app.run(port=5000, debug=True)

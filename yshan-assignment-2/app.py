from flask import Flask, render_template, jsonify, request
import numpy as np
from kmeans import KMeans

app = Flask(__name__)
kmeans = None
data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_data', methods=['POST'])
def set_data():
    global data
    data = request.json  # Expecting the data to be sent as JSON
    return jsonify({"message": "Data received."}), 200



@app.route('/run', methods=['POST'])
def run_kmeans():
    global kmeans, data
    
    # Get the JSON data from the request body
    req_data = request.get_json()

    # Handle the case where no data is sent or JSON decoding fails
    if not req_data:
        return jsonify({"error": "No data provided or invalid JSON."}), 400

    # Verify if the 'data' key exists in the request
    if 'data' not in req_data:
        return jsonify({"error": "'data' key missing in request."}), 400

    # Convert the data points from the request to a NumPy array
    data = np.array(req_data['data'])
    method = req_data.get('init_method', 'random')  # Use 'random' as default if not provided

    if data is None or len(data) == 0:
        return jsonify({"error": "Data not generated yet."}), 400
    
    kmeans = KMeans(k=3, init_method=method)
    clusters = kmeans.run_kmeans(data)
    
    return jsonify({
        'centroids': kmeans.centroids.tolist(),
        'clusters': clusters.tolist()
    })


if __name__ == '__main__':
    app.run(host='localhost', port=3000)

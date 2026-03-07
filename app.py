import os
import uuid
import warnings
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from config import UPLOADS_DIR, CLASSES, DISEASE_INFO, DEFAULT_INFO
from predictor import predictor
from report_generator import report_generator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

warnings.filterwarnings('ignore')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/classes')
def get_classes():
    return jsonify({"classes": CLASSES})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        scan_id = str(uuid.uuid4())
        ext = os.path.splitext(filename)[1]
        filepath = os.path.join(UPLOADS_DIR, f"{scan_id}{ext}")
        file.save(filepath)
        
        try:
            raw_results, prep_path = predictor.predict(filepath)
            top_predictions = predictor.get_top_predictions(raw_results)
            uncertainty = predictor.get_uncertainty(top_predictions)
            
            high_risk = any(p['risk_level'] == 'high' for p in top_predictions)
            
            top_class = top_predictions[0]['class']
            medical_info = DISEASE_INFO.get(top_class, DEFAULT_INFO)
            
            report_path = report_generator.generate_report(
                scan_id=scan_id,
                predictions=top_predictions,
                original_image_path=filepath,
                prep_image_path=prep_path,
                uncertainty=uncertainty,
                high_risk_alert=high_risk,
                medical_info=medical_info
            )
            
            return jsonify({
                'scan_id': scan_id,
                'top_prediction': top_predictions[0],
                'predictions': top_predictions,
                'uncertainty': uncertainty,
                'high_risk': high_risk,
                'medical_info': medical_info,
                'report_url': f"/report/{scan_id}"
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/report/<scan_id>')
def get_report(scan_id):
    report_path = os.path.join(app.root_path, 'reports', f"report_{scan_id}.pdf")
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True, download_name=f"skin_analysis_{scan_id}.pdf")
    return jsonify({'error': 'Report not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)

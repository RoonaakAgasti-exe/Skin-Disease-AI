from fpdf import FPDF
import os
from datetime import datetime
from config import REPORTS_DIR

class SkinDiseaseReportGenerator:
    def __init__(self):
        self.pdf = None
    
    def generate_report(self, scan_id, predictions, original_image_path, prep_image_path, uncertainty, high_risk_alert, medical_info, top_k=5):
        self.pdf = FPDF()
        self.pdf.add_page()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Skin Disease AI Clinical Analysis Report', 0, 1, 'C')
        self.pdf.ln(5)
        self.pdf.set_draw_color(0, 86, 179)
        self.pdf.line(10, 25, 200, 25)
        self.pdf.ln(10)
        
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, 'Scan Information', 0, 1)
        self.pdf.set_font('Arial', '', 10)
        self.pdf.cell(50, 6, 'Scan ID:', 0, 0)
        self.pdf.cell(0, 6, scan_id, 0, 1)
        self.pdf.cell(50, 6, 'Date:', 0, 0)
        self.pdf.cell(0, 6, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0, 1)
        self.pdf.cell(50, 6, 'Model:', 0, 0)
        self.pdf.cell(0, 6, 'Zero-Shot CLIP-L-14 CV-Ensemble', 0, 1)
        self.pdf.ln(5)

        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.set_fill_color(225, 245, 254)
        self.pdf.cell(0, 8, f"Clinical Overview: {predictions[0]['class']}", 0, 1, 'L', True)
        self.pdf.set_font('Arial', '', 10)
        self.pdf.multi_cell(0, 6, medical_info.get('description', 'No description available.'))
        self.pdf.ln(3)

        self.pdf.set_font('Arial', 'B', 11)
        self.pdf.cell(0, 7, 'Recommended Medicines:', 0, 1)
        self.pdf.set_font('Arial', '', 10)
        self.pdf.multi_cell(0, 6, ", ".join(medical_info.get('medicines', [])))
        self.pdf.ln(2)

        self.pdf.set_font('Arial', 'B', 11)
        self.pdf.cell(0, 7, 'Preventive Measures:', 0, 1)
        self.pdf.set_font('Arial', '', 10)
        self.pdf.multi_cell(0, 6, ", ".join(medical_info.get('preventives', [])))
        self.pdf.ln(5)
        
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, 'Top Predictions', 0, 1)
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.set_fill_color(240, 240, 240)
        self.pdf.cell(10, 8, 'Rank', 1, 0, 'C', True)
        self.pdf.cell(60, 8, 'Condition', 1, 0, 'C', True)
        self.pdf.cell(35, 8, 'Confidence', 1, 0, 'C', True)
        self.pdf.cell(35, 8, 'Risk Level', 1, 1, 'C', True)
        
        self.pdf.set_font('Arial', '', 9)
        for i, pred in enumerate(predictions[:top_k], 1):
            if i % 2 == 0:
                self.pdf.set_fill_color(250, 250, 250)
            else:
                self.pdf.set_fill_color(255, 255, 255)
            risk_color = 'High' if pred['risk_level'] == 'high' else 'Low'
            confidence_percent = f"{pred['confidence']*100:.1f}%"
            self.pdf.cell(10, 7, str(i), 1, 0, 'C', True)
            self.pdf.cell(60, 7, pred['class'], 1, 0, 'L', True)
            self.pdf.cell(35, 7, confidence_percent, 1, 0, 'C', True)
            self.pdf.cell(35, 7, risk_color, 1, 1, 'C', True)
        self.pdf.ln(5)
        
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, 'Analysis Metrics', 0, 1)
        self.pdf.set_font('Arial', '', 10)
        self.pdf.cell(50, 6, 'Model Uncertainty:', 0, 0)
        
        if uncertainty < 0.2:
            uncertainty_level = "Low"
        elif uncertainty < 0.4:
            uncertainty_level = "Moderate"
        elif uncertainty < 0.6:
            uncertainty_level = "High"
        else:
            uncertainty_level = "Very High"
            
        self.pdf.cell(0, 6, f"{uncertainty:.3f} ({uncertainty_level})", 0, 1)
        
        if high_risk_alert:
            self.pdf.set_text_color(255, 0, 0)
            self.pdf.cell(0, 6, 'WARNING: HIGH-RISK CONDITION DETECTED', 0, 1)
            self.pdf.set_text_color(0, 0, 0)
        else:
            self.pdf.cell(0, 6, '[OK] No high-risk conditions detected', 0, 1)
        self.pdf.ln(5)
        
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, 'Visual Analysis', 0, 1)
        self.pdf.ln(2)
        if os.path.exists(original_image_path):
            self.pdf.set_font('Arial', 'B', 10)
            self.pdf.cell(0, 6, 'Original Image:', 0, 1)
            self.pdf.image(original_image_path, x=10, y=self.pdf.get_y(), w=80)
            self.pdf.ln(60)
            
        if os.path.exists(prep_image_path):
            self.pdf.set_font('Arial', 'B', 10)
            self.pdf.cell(0, 6, 'CV Preprocessed Image:', 0, 1)
            self.pdf.image(prep_image_path, x=10, y=self.pdf.get_y(), w=80)
            self.pdf.ln(60)

        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, 'Important Disclaimer', 0, 1)
        self.pdf.set_font('Arial', '', 9)
        self.pdf.multi_cell(0, 5, "This AI-based analysis is provided for educational and decision support purposes only. The results should not be considered as a definitive medical diagnosis. Always consult with a qualified healthcare professional for proper diagnosis and treatment. The accuracy of this system may vary, and false positives/negatives are possible.")
        self.pdf.ln(5)
        self.pdf.set_font('Arial', 'I', 8)
        self.pdf.cell(0, 5, 'Generated by Skin Disease AI System v3.0 CV-Ensemble', 0, 1, 'R')
        
        report_path = os.path.join(REPORTS_DIR, f"report_{scan_id}.pdf")
        self.pdf.output(report_path)
        return report_path

report_generator = SkinDiseaseReportGenerator()
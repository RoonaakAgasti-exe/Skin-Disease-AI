from fpdf import FPDF
import os
import cv2
import numpy as np
from datetime import datetime
from config import *

class SkinDiseaseReportGenerator:
    def __init__(self):
        self.pdf = None
    
    def generate_report(self, scan_id, predictions, original_image_path, gradcam_image_path, 
                       uncertainty, high_risk_alert, top_k=5):
        """Generate comprehensive PDF report"""
        self.pdf = FPDF()
        self.pdf.add_page()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add title
        self._add_header()
        
        # Add patient/scanner information
        self._add_scan_info(scan_id)
        
        # Add prediction results
        self._add_prediction_results(predictions[:top_k])
        
        # Add uncertainty analysis
        self._add_uncertainty_analysis(uncertainty, high_risk_alert)
        
        # Add visualizations
        self._add_visualizations(original_image_path, gradcam_image_path)
        
        # Add medical disclaimer
        self._add_disclaimer()
        
        # Save report
        report_path = os.path.join(REPORTS_DIR, f"report_{scan_id}.pdf")
        self.pdf.output(report_path)
        
        return report_path
    
    def _add_header(self):
        """Add report header"""
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Skin Disease AI Analysis Report', 0, 1, 'C')
        self.pdf.ln(5)
        
        # Add line
        self.pdf.set_draw_color(128, 128, 128)
        self.pdf.line(10, 25, 200, 25)
        self.pdf.ln(10)
    
    def _add_scan_info(self, scan_id):
        """Add scan information"""
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, 'Scan Information', 0, 1)
        self.pdf.set_font('Arial', '', 10)
        
        self.pdf.cell(50, 6, 'Scan ID:', 0, 0)
        self.pdf.cell(0, 6, scan_id, 0, 1)
        
        self.pdf.cell(50, 6, 'Date:', 0, 0)
        self.pdf.cell(0, 6, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0, 1)
        
        self.pdf.cell(50, 6, 'Model Version:', 0, 0)
        self.pdf.cell(0, 6, 'Ensemble v1.0', 0, 1)
        
        self.pdf.ln(5)
    
    def _add_prediction_results(self, predictions):
        """Add prediction results table"""
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, 'Top Predictions', 0, 1)
        
        # Table header
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.set_fill_color(240, 240, 240)
        self.pdf.cell(10, 8, 'Rank', 1, 0, 'C', True)
        self.pdf.cell(60, 8, 'Condition', 1, 0, 'C', True)
        self.pdf.cell(35, 8, 'Confidence', 1, 0, 'C', True)
        self.pdf.cell(35, 8, 'Risk Level', 1, 1, 'C', True)
        
        # Table rows
        self.pdf.set_font('Arial', '', 9)
        for i, pred in enumerate(predictions, 1):
            # Alternate row colors
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
    
    def _add_uncertainty_analysis(self, uncertainty, high_risk_alert):
        """Add uncertainty analysis"""
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, 'Analysis Metrics', 0, 1)
        self.pdf.set_font('Arial', '', 10)
        
        # Uncertainty
        self.pdf.cell(50, 6, 'Model Uncertainty:', 0, 0)
        uncertainty_level = self._get_uncertainty_level(uncertainty)
        self.pdf.cell(0, 6, f"{uncertainty:.3f} ({uncertainty_level})", 0, 1)
        
        # Risk alert
        if high_risk_alert:
            self.pdf.set_text_color(255, 0, 0)
            self.pdf.cell(0, 6, 'WARNING: HIGH-RISK CONDITION DETECTED - URGENT MEDICAL EVALUATION RECOMMENDED', 0, 1)
            self.pdf.set_text_color(0, 0, 0)
        else:
            self.pdf.cell(0, 6, '[OK] No high-risk conditions detected', 0, 1)
        
        self.pdf.ln(5)
    
    def _add_visualizations(self, original_path, gradcam_path):
        """Add image visualizations"""
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, 'Visual Analysis', 0, 1)
        self.pdf.ln(2)
        
        # Try to add images
        try:
            # Original image
            if os.path.exists(original_path):
                self.pdf.set_font('Arial', 'B', 10)
                self.pdf.cell(0, 6, 'Original Image:', 0, 1)
                self.pdf.image(original_path, x=10, y=self.pdf.get_y(), w=80)
                self.pdf.ln(60)  # Move down for next image
            
            # GradCAM visualization
            if os.path.exists(gradcam_path):
                self.pdf.set_font('Arial', 'B', 10)
                self.pdf.cell(0, 6, 'GradCAM Visualization:', 0, 1)
                self.pdf.image(gradcam_path, x=10, y=self.pdf.get_y(), w=80)
                self.pdf.ln(60)
                
        except Exception as e:
            self.pdf.set_font('Arial', 'I', 10)
            self.pdf.cell(0, 6, f'Image visualization unavailable: {str(e)}', 0, 1)
            self.pdf.ln(10)
    
    def _add_disclaimer(self):
        """Add medical disclaimer"""
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, 'Important Disclaimer', 0, 1)
        self.pdf.set_font('Arial', '', 9)
        
        disclaimer_text = (
            "This AI-based analysis is provided for educational and decision support purposes only. "
            "The results should not be considered as a definitive medical diagnosis. "
            "Always consult with a qualified healthcare professional for proper diagnosis and treatment. "
            "The accuracy of this system may vary, and false positives/negatives are possible."
        )
        
        self.pdf.multi_cell(0, 5, disclaimer_text)
        self.pdf.ln(5)
        
        # Add signature line
        self.pdf.set_font('Arial', 'I', 8)
        self.pdf.cell(0, 5, 'Generated by Skin Disease AI System v1.0', 0, 1, 'R')
    
    def _get_uncertainty_level(self, uncertainty):
        """Convert uncertainty value to descriptive level"""
        if uncertainty < 0.2:
            return "Low"
        elif uncertainty < 0.4:
            return "Moderate"
        elif uncertainty < 0.6:
            return "High"
        else:
            return "Very High"

# Global report generator instance
report_generator = SkinDiseaseReportGenerator()
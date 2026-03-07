import os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline
from config import CLASSES, HIGH_RISK, UPLOADS_DIR

class AdvancedSkinPredictor:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "zero-shot-image-classification", 
            model="openai/clip-vit-large-patch14", 
            device=self.device
        )
        self.prompts = [
            "a dermatological photo of {}"
        ]

    def remove_hair(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        inpainted = cv2.inpaint(img, threshold, 1, cv2.INPAINT_TELEA)
        return inpainted

    def color_constancy(self, img):
        b, g, r = cv2.split(img)
        avg_b = np.mean(b)
        avg_g = np.mean(g)
        avg_r = np.mean(r)
        avg_gray = (avg_b + avg_g + avg_r) / 3
        
        b = np.clip(b * (avg_gray / (avg_b + 1e-5)), 0, 255).astype(np.uint8)
        g = np.clip(g * (avg_gray / (avg_g + 1e-5)), 0, 255).astype(np.uint8)
        r = np.clip(r * (avg_gray / (avg_r + 1e-5)), 0, 255).astype(np.uint8)
        
        return cv2.merge([b, g, r])

    def preprocess_image(self, image_path):
        hair_removed = self.remove_hair(image_path)
        color_corrected = self.color_constancy(hair_removed)
        
        prep_path = image_path.replace('.jpg', '_prep.jpg').replace('.png', '_prep.png').replace('.jpeg', '_prep.jpeg')
        if prep_path == image_path:
            prep_path = image_path + "_prep.jpg"
            
        cv2.imwrite(prep_path, color_corrected)
        return prep_path

    def predict(self, image_path):
        prep_path = self.preprocess_image(image_path)
        image = Image.open(prep_path).convert("RGB")
        
        ensemble_scores = {c: 0.0 for c in CLASSES}
        
        for prompt in self.prompts:
            candidate_labels = [prompt.format(c.replace("_", " ")) for c in CLASSES]
            results = self.classifier(image, candidate_labels=candidate_labels)
            
            if isinstance(results, dict):
                for label, score in zip(results['labels'], results['scores']):
                    for c in CLASSES:
                        if c.replace("_", " ").lower() in label.lower():
                            ensemble_scores[c] += score
                            break
            else:
                for res in results:
                    for c in CLASSES:
                        if c.replace("_", " ").lower() in res['label'].lower():
                            ensemble_scores[c] += res['score']
                            break

        for c in CLASSES:
            ensemble_scores[c] /= len(self.prompts)

        sorted_results = [{"class": k, "score": v} for k, v in sorted(ensemble_scores.items(), key=lambda item: item[1], reverse=True)]
        return sorted_results, prep_path

    def get_top_predictions(self, sorted_results, top_k=5):
        top_results = []
        for i in range(min(top_k, len(sorted_results))):
            class_name = sorted_results[i]['class']
            confidence = sorted_results[i]['score']
            
            is_high_risk = class_name in HIGH_RISK and confidence > 0.3
            top_results.append({
                'class': class_name,
                'confidence': float(confidence),
                'risk_level': 'high' if is_high_risk else 'low'
            })
        return top_results

    def get_uncertainty(self, top_results):
        if not top_results:
            return 1.0
        
        scores = np.array([res['confidence'] for res in top_results])
        if len(scores) < 2: return max(0.0, 1.0 - scores[0])
        
        entropy = -np.sum(scores * np.log(scores + 1e-7))
        normalized_entropy = entropy / np.log(len(CLASSES))
        
        top_conf = scores[0]
        if top_conf > 0.7:
            normalized_entropy *= 0.2
        elif top_conf > 0.4:
            normalized_entropy *= 0.5
            
        return float(min(1.0, max(0.0, normalized_entropy)))

predictor = AdvancedSkinPredictor()
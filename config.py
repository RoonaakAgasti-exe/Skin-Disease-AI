import os

CLASSES = [
    "Acne", "Actinic_Keratosis", "Benign_tumors", "Bullous", "Candidiasis",
    "DrugEruption", "Eczema", "Infestations_Bites", "Lichen", "Lupus",
    "Moles", "Psoriasis", "Rosacea", "Seborrh_Keratoses", "SkinCancer",
    "Sun_Sunlight_Damage", "Tinea", "Unknown_Normal", "Vascular_Tumors",
    "Vasculitis", "Vitiligo", "Warts"
]

HIGH_RISK = {"SkinCancer", "Actinic_Keratosis"}

DISEASE_INFO = {
    "Acne": {
        "medicines": ["Benzoyl Peroxide", "Salicylic Acid", "Retinoids", "Antibiotics"],
        "preventives": ["Regular cleansing", "Non-comedogenic makeup", "Healthy diet", "Stress management"],
        "description": "A common skin condition that occurs when hair follicles become plugged with oil and dead skin cells."
    },
    "Actinic_Keratosis": {
        "medicines": ["5-Fluorouracil", "Imiquimod", "Diclofenac gel"],
        "preventives": ["Sun protection", "Regular skin checks", "Avoid tanning beds"],
        "description": "A rough, scaly patch on the skin that develops from years of sun exposure."
    },
    "SkinCancer": {
        "medicines": ["Specific to cancer type - REQUIRES ONCOLOGIST CONSULTATION"],
        "preventives": ["Strict sun protection", "Regular mole mapping", "Wide-brimmed hats"],
        "description": "The abnormal growth of skin cells, most often develops on skin exposed to the sun."
    },
    "Eczema": {
        "medicines": ["Corticosteroids", "Calcineurin inhibitors", "Antihistamines"],
        "preventives": ["Moisturizing daily", "Avoiding triggers", "Mild soaps"],
        "description": "A condition that makes your skin red and itchy."
    },
    "Psoriasis": {
        "medicines": ["Topical corticosteroids", "Vitamin D analogues", "Biologics"],
        "preventives": ["Stress reduction", "Avoiding skin injuries", "Moisturizing"],
        "description": "A condition in which skin cells build up and form scales and itchy, dry patches."
    },
    "Rosacea": {
        "medicines": ["Metronidazole", "Azelaic acid", "Brimonidine"],
        "preventives": ["Avoiding spicy foods", "Sun protection", "Gentle skincare"],
        "description": "A condition that causes redness and often small, red, pus-filled bumps on the face."
    },
    "Benign_tumors": {
        "medicines": ["Observation", "Excision if symptomatic"],
        "preventives": ["Skin self-exams", "UV protection"],
        "description": "Non-cancerous skin growths such as seborrheic keratoses or dermatofibromas."
    },
    "Bullous": {
        "medicines": ["Corticosteroids", "Immunosuppressants"],
        "preventives": ["Avoid skin trauma", "Wound care"],
        "description": "Blistering skin disorders like pemphigoid or pemphigus."
    },
    "Candidiasis": {
        "medicines": ["Clotrimazole", "Nystatin", "Fluconazole"],
        "preventives": ["Keep skin dry", "Weight management", "Blood sugar control"],
        "description": "A fungal infection caused by yeast, typically in skin folds."
    },
    "DrugEruption": {
        "medicines": ["Antihistamines", "Corticosteroids"],
        "preventives": ["Avoid known allergen drugs", "Carry medical alert ID"],
        "description": "An adverse skin reaction to a medication."
    },
    "Infestations_Bites": {
        "medicines": ["Permethrin", "Ivermectin", "Calamine"],
        "preventives": ["Insect repellent", "Bug nets", "Proper clothing"],
        "description": "Skin surface reactions to insect bites or parasitic infestations like scabies or lice."
    },
    "Lichen": {
        "medicines": ["Topical steroids", "Retinoids", "Light therapy"],
        "preventives": ["Stress management", "Avoid skin injury"],
        "description": "An inflammatory condition that affects skin, hair, and nails (e.g., Lichen Planus)."
    },
    "Lupus": {
        "medicines": ["Hydroxychloroquine", "Corticosteroids"],
        "preventives": ["Strict sun protection", "Smoking cessation"],
        "description": "An autoimmune disease that can cause a characteristic butterfly rash or discoid lesions."
    },
    "Seborrh_Keratoses": {
        "medicines": ["Cryotherapy", "Curettage", "Laser therapy"],
        "preventives": ["Sun protection", "General skin health"],
        "description": "Common non-cancerous skin growth that often looks like a waxy or pasted-on spot."
    },
    "Sun_Sunlight_Damage": {
        "medicines": ["Tretinoin", "Vitamin C serum", "Chemical peels"],
        "preventives": ["Broad-spectrum SPF 50", "Protective clothing", "Avoid peak sun"],
        "description": "Skin changes like photoaging or solar lentigines caused by chronic UV exposure."
    },
    "Tinea": {
        "medicines": ["Terbinafine", "Ketoconazole", "Miconazole"],
        "preventives": ["Dry skin thoroughly", "Avoid sharing towels", "Wear breathable fabrics"],
        "description": "A group of fungal infections including ringworm, athlete's foot, and jock itch."
    },
    "Unknown_Normal": {
        "medicines": ["None required"],
        "preventives": ["Maintain basic hygiene", "Regular moisturizing"],
        "description": "Skin appears clinically normal or has no identifiable pathological patterns."
    },
    "Vascular_Tumors": {
        "medicines": ["Beta-blockers", "Laser therapy", "Sclerotherapy"],
        "preventives": ["Trauma prevention to the area"],
        "description": "Growths made of blood vessels, such as hemangiomas or port-wine stains."
    },
    "Vasculitis": {
        "medicines": ["Corticosteroids", "Colchicine"],
        "preventives": ["Identify and avoid triggers", "Rest during flare-ups"],
        "description": "Inflammation of the blood vessels in the skin, often presenting as purple spots (purpura)."
    },
    "Warts": {
        "medicines": ["Salicylic acid", "Cryotherapy", "Cantharidin"],
        "preventives": ["Avoid touching warts", "Foot hygiene", "Avoid sharing personal items"],
        "description": "Small, fleshy bumps on the skin or mucous membranes caused by HPV."
    }
}

DEFAULT_INFO = {
    "medicines": ["Consult a dermatologist for targeted therapy"],
    "preventives": ["Maintain skin hygiene", "Use SPF 30+ sunscreen", "Keep skin hydrated"],
    "description": "Dermatological condition requiring clinical evaluation."
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
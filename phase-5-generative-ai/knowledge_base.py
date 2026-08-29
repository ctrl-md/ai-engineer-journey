"""
A tiny clinical knowledge base -- short documents standing in for
clinical guideline excerpts. Deliberately small and simple so the
RAG mechanics (chunking, embedding, retrieval) stay clear.
"""

documents = [
    "Hypertension is diagnosed when blood pressure consistently exceeds 130/80 mmHg. First-line treatment usually includes lifestyle changes and, if needed, ACE inhibitors or thiazide diuretics.",
    "Type 2 diabetes management centers on blood glucose control. Metformin is typically the first-line medication, alongside diet and exercise changes.",
    "Patients with chronic kidney disease require dose adjustments for many medications, since reduced kidney function slows drug clearance from the body.",
    "Fatigue is a common but nonspecific symptom. Common causes include anemia, hypothyroidism, poor sleep, depression, and viral infections.",
    "Pneumonia symptoms include cough, fever, and shortness of breath. Diagnosis is often confirmed with a chest X-ray showing lung opacity.",
    "Aspirin should be used cautiously in patients with a history of gastrointestinal bleeding, due to increased bleeding risk.",
]

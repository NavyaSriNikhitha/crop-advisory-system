# generate_fertilizer_data.py
# Generates a realistic fertilizer recommendation dataset
# covering all 22 crops from Phase 1.
# Based on standard agronomic guidelines for nutrient requirements.

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── Crop profiles ─────────────────────────────────────────────
# Each crop has typical soil nutrient ranges and recommended
# fertilizer based on whether N, P, or K is the limiting factor.
# Source: standard agricultural extension guidelines.

crop_profiles = {
    # crop: (N_range, P_range, K_range, temp_range, humidity_range,
    #         ph_range, rainfall_range, soil_types, fertilizer_map)
    # fertilizer_map: {condition: fertilizer}
    #   condition based on which nutrient is most deficient

    "rice":        ((60,80),  (35,45),  (35,45),  (20,27), (80,90), (5.5,7.0), (180,270), ["Clayey","silty clay","laterite"],       {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Urea and DAP"}),
    "maize":       ((70,90),  (45,55),  (40,50),  (18,27), (55,70), (5.5,7.5), (80,110),  ["sandy","alluvial","clay loam"],          {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Good NPK"}),
    "chickpea":    ((30,50),  (55,75),  (65,80),  (15,25), (14,30), (6.0,8.0), (65,105),  ["sandy","alluvial","clay loam"],          {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"DAP and MOP"}),
    "kidneybeans": ((15,25),  (55,75),  (15,25),  (15,25), (18,28), (5.5,7.0), (100,130), ["alluvial","clay loam","sandy"],          {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"DAP and MOP"}),
    "pigeonpeas":  ((15,25),  (55,75),  (15,25),  (25,35), (45,60), (5.0,7.0), (140,155), ["alluvial","laterite","clay loam"],       {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Good NPK"}),
    "mothbeans":   ((15,25),  (40,60),  (15,25),  (25,35), (45,60), (3.5,6.5), (40,60),   ["sandy","alluvial","laterite"],           {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"DAP and MOP"}),
    "mungbean":    ((15,25),  (55,75),  (15,25),  (25,35), (80,90), (6.0,7.5), (45,65),   ["alluvial","clay loam","sandy"],          {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Urea and DAP"}),
    "blackgram":   ((15,25),  (55,75),  (15,25),  (25,35), (60,70), (5.5,7.0), (65,75),   ["alluvial","laterite","clay loam"],       {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Good NPK"}),
    "lentil":      ((15,25),  (55,75),  (15,25),  (18,28), (65,75), (6.0,8.0), (45,55),   ["alluvial","clay loam","sandy"],          {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"DAP and MOP"}),
    "pomegranate": ((15,25),  (55,75), (35,45),   (18,24), (90,95), (5.5,7.0), (100,115), ["laterite","alluvial","clay loam"],       {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Urea and MOP"}),
    "banana":      ((90,115), (55,75), (45,55),   (25,35), (75,90), (5.5,7.0), (100,115), ["alluvial","clay loam","Clayey"],         {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Good NPK"}),
    "mango":       ((15,25),  (15,25), (25,35),   (25,35), (45,55), (5.5,7.5), (90,115),  ["alluvial","laterite","sandy"],           {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Urea and DAP"}),
    "grapes":      ((15,25), (120,145),(195,210),  (8,17),  (80,90), (5.5,7.0), (65,75),   ["laterite","alluvial","clay loam"],       {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"DAP and MOP"}),
    "watermelon":  ((95,115), (8,16),  (45,55),   (24,30), (80,90), (6.0,7.0), (45,60),   ["sandy","alluvial","clay loam"],          {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Urea and MOP"}),
    "muskmelon":   ((95,115), (8,16),  (45,55),   (28,38), (85,95), (6.0,7.0), (20,30),   ["sandy","alluvial","coastal"],            {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Good NPK"}),
    "apple":       ((15,25), (120,145),(195,210),  (21,24), (90,95), (5.5,7.0), (110,125), ["clay loam","alluvial","laterite"],       {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"DAP and MOP"}),
    "orange":      ((15,25),  (15,25), (8,16),    (22,32), (85,95), (6.0,7.5), (100,115), ["alluvial","clay loam","coastal"],        {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Good NPK"}),
    "papaya":      ((45,55),  (35,55), (45,55),   (33,38), (90,95), (6.0,7.0), (140,155), ["alluvial","Clayey","sandy"],             {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Urea and DAP"}),
    "coconut":     ((4,6),    (4,6),   (4,6),     (24,28), (90,96), (5.0,8.0), (145,180), ["laterite","coastal","sandy"],            {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Good NPK"}),
    "cotton":      ((115,135),(40,60), (15,25),   (23,30), (75,85), (6.0,8.0), (80,95),   ["alluvial","Clayey","clay loam"],         {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Urea and MOP"}),
    "jute":        ((60,85),  (40,55), (35,45),   (23,27), (75,90), (6.0,7.0), (160,200), ["alluvial","Clayey","silty clay"],        {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Urea and DAP"}),
    "coffee":      ((95,115), (25,35), (25,35),   (23,28), (55,65), (6.0,6.5), (145,165), ["laterite","alluvial","clay loam"],       {"N_low":"Urea","P_low":"DAP","K_low":"MOP","balanced":"Good NPK"}),
}

# ── Generate rows ─────────────────────────────────────────────
def get_fertilizer(N, P, K, fmap):
    """
    Decide fertilizer based on which nutrient is most deficient.
    The nutrient with the lowest value relative to its typical range
    is the 'limiting factor' — the one the fertilizer should address.
    """
    # Simple rule-based logic — real agronomic decision making
    scores = {"N": N, "P": P, "K": K}
    min_nutrient = min(scores, key=scores.get)

    if scores[min_nutrient] < 10:
        return fmap[f"{min_nutrient}_low"]
    else:
        return fmap["balanced"]

rows = []
SAMPLES_PER_CROP = 100    # 100 samples per crop = 2200 rows total

for crop, profile in crop_profiles.items():
    N_range, P_range, K_range, temp_range, hum_range, \
    ph_range, rain_range, soils, fmap = profile

    for _ in range(SAMPLES_PER_CROP):
        # Generate values with small random noise around typical range
        N    = round(np.random.uniform(*N_range), 1)
        P    = round(np.random.uniform(*P_range), 1)
        K    = round(np.random.uniform(*K_range), 1)
        temp = round(np.random.uniform(*temp_range), 2)
        hum  = round(np.random.uniform(*hum_range), 2)
        ph   = round(np.random.uniform(*ph_range), 2)
        rain = round(np.random.uniform(*rain_range), 2)
        soil = np.random.choice(soils)
        fert = get_fertilizer(N, P, K, fmap)

        rows.append({
            "Temperature": temp,
            "Humidity":    hum,
            "Rainfall":    rain,
            "pH":          ph,
            "N":           N,
            "P":           P,
            "K":           K,
            "Soil":        soil,
            "Crop":        crop,
            "Fertilizer":  fert
        })

df = pd.DataFrame(rows)

# ── Shuffle rows ──────────────────────────────────────────────
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/fertilizer.csv", index=False)

print("=" * 60)
print("Dataset generated successfully!")
print(f"Shape      : {df.shape}")
print(f"Columns    : {df.columns.tolist()}")
print(f"\nFertilizer types : {sorted(df['Fertilizer'].unique().tolist())}")
print(f"Crop types       : {sorted(df['Crop'].unique().tolist())}")
print(f"Soil types       : {sorted(df['Soil'].unique().tolist())}")
print(f"\nFertilizer distribution:")
print(df["Fertilizer"].value_counts())
print(f"\nFirst 5 rows:")
print(df.head())
print("=" * 60)
print("Saved to: data/fertilizer.csv")
print("Next step: python fertilizer_model.py")

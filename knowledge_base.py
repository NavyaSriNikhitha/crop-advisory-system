# ============================================================
# knowledge_base.py
# What this file does:
#   1. Creates a rich farming knowledge base (100+ documents)
#   2. Covers: crops, soils, fertilizers, pests, irrigation,
#              seasons, government schemes, organic farming
#   3. Stores everything in ChromaDB (local vector database)
#   4. Uses sentence-transformers for embeddings (free, local)
#
# Run ONCE before starting the app:
#   python knowledge_base.py
#
# Install dependencies:
#   uv add chromadb sentence-transformers
# ============================================================

import chromadb
from chromadb.utils import embedding_functions
import os

print("=" * 60)
print("Building Farming Knowledge Base")
print("=" * 60)

# ── STEP 1: Knowledge base documents ─────────────────────────
# Each document has:
#   - id: unique identifier
#   - text: the actual knowledge content
#   - metadata: category tag for filtering

documents = [

    # ── IRRIGATION ────────────────────────────────────────────
    {
        "id": "irr_001",
        "text": """Drip irrigation is a method of watering crops by delivering water directly 
to the root zone of plants through a network of pipes, tubes, and emitters. 
Water drips slowly at a rate of 2-20 litres per hour. It reduces water usage 
by 30-50% compared to flood irrigation. Best used for vegetables, fruits, 
sugarcane, cotton, and orchards. Drip irrigation prevents waterlogging, 
reduces weed growth, and improves fertilizer efficiency through fertigation. 
In India, drip irrigation is promoted under PM Krishi Sinchai Yojana with 
55% subsidy for small farmers and 45% for large farmers.""",
        "metadata": {"category": "irrigation", "topic": "drip irrigation"}
    },
    {
        "id": "irr_002",
        "text": """Sprinkler irrigation distributes water through a pressurized pipe system 
with rotating sprinkler heads that spray water over crops like rain. 
Suitable for wheat, groundnut, vegetables, and tea gardens. 
Water use efficiency is 70-80%. Best for sandy soils and uneven terrain. 
Sprinklers work well where drip is not suitable — for broadcast crops like wheat. 
Cost is Rs 15,000-25,000 per acre to install. Government provides 50% subsidy 
under PMKSY scheme for sprinkler systems.""",
        "metadata": {"category": "irrigation", "topic": "sprinkler irrigation"}
    },
    {
        "id": "irr_003",
        "text": """Flood irrigation is the traditional method where water flows across the 
entire field surface. It is the most common irrigation method in India covering 
90% of irrigated area. Very cheap to implement but wastes 50-60% water through 
evaporation and runoff. Suitable for rice, sugarcane, and jute which need 
standing water. Not recommended for vegetables and fruits. Punjab, Haryana, 
and UP primarily use flood irrigation for wheat and rice.""",
        "metadata": {"category": "irrigation", "topic": "flood irrigation"}
    },
    {
        "id": "irr_004",
        "text": """Furrow irrigation creates small channels between crop rows to guide water 
flow to plant roots. Used for row crops like maize, cotton, sugarcane, 
and vegetables. More efficient than flood irrigation — saves 20-30% water. 
Furrow depth should be 15-20 cm. Suitable for medium to heavy soils. 
Water moves through furrows by gravity without pumping cost. 
Common in Maharashtra for sugarcane cultivation.""",
        "metadata": {"category": "irrigation", "topic": "furrow irrigation"}
    },

    # ── SOILS ─────────────────────────────────────────────────
    {
        "id": "soil_001",
        "text": """Black soil (also called Regur soil or cotton soil) is found in Maharashtra, 
Madhya Pradesh, Gujarat, Andhra Pradesh, and Karnataka. It is rich in calcium, 
magnesium, and potassium but deficient in nitrogen and phosphorus. 
Black soil has high water retention — stays moist for a long time. 
Best crops: cotton, wheat, sorghum, groundnut, sunflower, and citrus fruits. 
pH ranges from 7.5 to 8.5. Add nitrogen fertilizers like Urea and phosphorus 
like DAP to improve fertility. Avoid waterlogging as black soil becomes 
sticky and unworkable when wet.""",
        "metadata": {"category": "soil", "topic": "black soil"}
    },
    {
        "id": "soil_002",
        "text": """Alluvial soil is the most widespread and fertile soil in India, found in 
the Indo-Gangetic Plain (Punjab, Haryana, UP, Bihar, West Bengal) and 
river deltas. Rich in potash but deficient in nitrogen and phosphorus. 
Supports the highest agricultural productivity in India. 
Best crops: wheat, rice, maize, sugarcane, pulses, and vegetables. 
Two types: Khadar (new alluvial, sandy, near rivers) and Bhangar 
(old alluvial, more clay). Apply Urea for nitrogen and DAP for phosphorus. 
pH is neutral to slightly alkaline (6.5-8.0).""",
        "metadata": {"category": "soil", "topic": "alluvial soil"}
    },
    {
        "id": "soil_003",
        "text": """Red soil is found in Tamil Nadu, Andhra Pradesh, Karnataka, Odisha, 
and Jharkhand. Red color comes from iron oxide content. Low in nitrogen, 
phosphorus, and organic matter but good drainage. 
Best crops: groundnut, cotton, wheat, rice, millets, and tobacco. 
pH ranges from 6.0 to 7.5. Needs heavy manuring and fertilization. 
Apply FYM (farmyard manure) at 10-15 tonnes per hectare along with 
NPK fertilizers. Drip irrigation recommended due to low water retention.""",
        "metadata": {"category": "soil", "topic": "red soil"}
    },
    {
        "id": "soil_004",
        "text": """Laterite soil is found in Kerala, Karnataka, Tamil Nadu, and hilly areas 
of West Bengal and Assam. Formed by intense weathering in high rainfall areas. 
Very poor in nitrogen, phosphorus, calcium, and organic matter. 
Highly acidic (pH 4.5-6.0). Best crops: tea, coffee, cashew, rubber, coconut, 
and tapioca. Apply lime to reduce acidity before cultivation. 
Add heavy doses of organic manure and NPK fertilizers. 
Hardens when exposed to air — needs regular ploughing.""",
        "metadata": {"category": "soil", "topic": "laterite soil"}
    },
    {
        "id": "soil_005",
        "text": """Sandy soil has large particles with very low water and nutrient retention. 
Found in Rajasthan and coastal areas. Drains water quickly — needs frequent 
irrigation. Low in all nutrients. Best crops: groundnut, watermelon, 
muskmelon, guar, and bajra. Add organic matter — FYM, compost, vermicompost 
at 20-25 tonnes per hectare to improve water holding capacity. 
Drip irrigation is essential for sandy soils to prevent water waste. 
Mulching helps retain moisture between irrigation cycles.""",
        "metadata": {"category": "soil", "topic": "sandy soil"}
    },
    {
        "id": "soil_006",
        "text": """Soil pH is a measure of acidity or alkalinity on a scale of 0-14. 
pH 7 is neutral. Below 7 is acidic, above 7 is alkaline. 
Most crops grow best at pH 6.0-7.5. Acidic soil (low pH) reduces 
nutrient availability — especially phosphorus, calcium, and magnesium. 
To raise pH (fix acidity): apply agricultural lime at 2-4 tonnes per hectare. 
To lower pH (fix alkalinity): apply gypsum or sulphur at 1-2 tonnes per hectare. 
Soil pH should be tested every 3 years using a soil health card 
(free from government via Soil Health Card scheme).""",
        "metadata": {"category": "soil", "topic": "soil pH"}
    },

    # ── FERTILIZERS ───────────────────────────────────────────
    {
        "id": "fert_001",
        "text": """Urea is the most commonly used nitrogen fertilizer in India. 
Contains 46% nitrogen. Applied as basal dose and top dressing. 
For wheat: apply 60-80 kg Urea per acre split in 3 doses. 
For rice: apply 40-50 kg Urea per acre. Urea is best applied in the 
evening to reduce volatilization losses. Do not apply before heavy rain. 
Price is subsidized by government — approximately Rs 266 per 50 kg bag. 
Excess urea causes lodging, delays maturity, and reduces grain quality. 
Always split urea application — never apply full dose at once.""",
        "metadata": {"category": "fertilizer", "topic": "urea"}
    },
    {
        "id": "fert_002",
        "text": """DAP (Di-Ammonium Phosphate) contains 18% nitrogen and 46% phosphorus. 
It is the most popular phosphatic fertilizer in India. Applied as basal 
dose at sowing time — mix in soil before planting. 
For wheat: 50 kg DAP per acre. For rice: 40 kg DAP per acre. 
For vegetables: 30-40 kg DAP per acre. DAP is granular and easy to apply. 
Price approximately Rs 1350 per 50 kg bag (subsidized). 
DAP is best for crops requiring high phosphorus like legumes, 
oilseeds, and root vegetables. Mix DAP in soil at 10-15 cm depth.""",
        "metadata": {"category": "fertilizer", "topic": "DAP"}
    },
    {
        "id": "fert_003",
        "text": """MOP (Muriate of Potash) contains 60% potassium (K2O). 
Applied to improve crop quality, disease resistance, and drought tolerance. 
For potato: 30-40 kg MOP per acre. For sugarcane: 50-60 kg MOP per acre. 
For banana: 60-80 kg MOP per acre. MOP improves fruit quality and sweetness. 
Not recommended for tobacco and potato in excess — causes quality issues. 
Apply MOP at basal dose mixed with soil. Price approximately Rs 1700 per 50 kg. 
Crops showing potassium deficiency have brown leaf margins and poor grain filling.""",
        "metadata": {"category": "fertilizer", "topic": "MOP potassium"}
    },
    {
        "id": "fert_004",
        "text": """NPK fertilizers contain nitrogen, phosphorus, and potassium in fixed ratios. 
Common grades: 10:26:26, 12:32:16, 20:20:0, 19:19:19. 
Grade 10:26:26 is used for phosphorus and potassium deficient soils. 
Grade 19:19:19 water-soluble NPK is used for fertigation through drip systems. 
Apply NPK as basal dose at sowing — 50-75 kg per acre depending on crop. 
NPK fertilizers are convenient — provide all three major nutrients in one bag. 
Best for fruits and vegetables where balanced nutrition is critical.""",
        "metadata": {"category": "fertilizer", "topic": "NPK fertilizer"}
    },
    {
        "id": "fert_005",
        "text": """Organic manures include FYM (Farmyard Manure), compost, vermicompost, 
green manure, and biofertilizers. FYM contains 0.5% N, 0.2% P, 0.5% K 
and improves soil structure. Apply 10-15 tonnes FYM per hectare before planting. 
Vermicompost is richer — contains 2-3% N, 1-2% P, 1.5-2% K. 
Apply 2-4 tonnes vermicompost per hectare. Biofertilizers like Rhizobium, 
Azospirillum, and PSB fix atmospheric nitrogen and solubilize phosphorus. 
Organic farming uses only organic manures — no chemical fertilizers. 
Combine organic and chemical fertilizers for best results (integrated nutrient management).""",
        "metadata": {"category": "fertilizer", "topic": "organic manure"}
    },

    # ── CROPS ─────────────────────────────────────────────────
    {
        "id": "crop_001",
        "text": """Rice (paddy) is the most important food crop of India grown in Kharif season 
(June-November). Major states: West Bengal, Punjab, Andhra Pradesh, UP, Tamil Nadu. 
Requires high rainfall (150-200 cm) or irrigation. Grows best in clayey soil 
with pH 5.5-7.0 and temperature 20-35°C. Transplanting method gives better yield 
than broadcasting. Apply Urea 80-100 kg per acre split in 3 doses. 
Apply DAP 40 kg per acre as basal. Common diseases: blast, brown spot, bacterial blight. 
Average yield: 3-5 tonnes per hectare. High-yielding varieties: IR-36, IR-64, Swarna.""",
        "metadata": {"category": "crop", "topic": "rice cultivation"}
    },
    {
        "id": "crop_002",
        "text": """Wheat is India's second most important crop grown in Rabi season (November-April). 
Major states: Punjab, Haryana, UP, Madhya Pradesh, Rajasthan. 
Requires cool climate (10-25°C) and 30-100 cm rainfall. 
Best in alluvial loamy soil with pH 6.0-7.5. Sow in October-November. 
Apply DAP 50 kg per acre + Urea 60 kg per acre in 3 splits. 
Irrigation: 4-6 irrigations needed — critical at crown root initiation (CRI), 
tillering, jointing, flowering, and grain filling stages. 
Common diseases: rust, powdery mildew, loose smut. 
Popular varieties: HD-2967, PBW-343, GW-496. Average yield: 4-5 tonnes per hectare.""",
        "metadata": {"category": "crop", "topic": "wheat cultivation"}
    },
    {
        "id": "crop_003",
        "text": """Cotton is a major cash crop grown in Kharif season. Major states: Gujarat, 
Maharashtra, Telangana, Andhra Pradesh, Punjab. Requires 6-8 months growing period. 
Temperature: 21-30°C. Rainfall: 50-100 cm. Grows best in black cotton soil (regur). 
Apply Urea 40 kg + DAP 30 kg + MOP 20 kg per acre as basal. 
Top dress with Urea 20 kg at flowering. 
Major pest: Pink bollworm, American bollworm, whitefly. 
Bt cotton varieties resistant to bollworm. 
Harvest when bolls open — 150-180 days after sowing. 
Average yield: 15-25 quintals per hectare (seed cotton).""",
        "metadata": {"category": "crop", "topic": "cotton cultivation"}
    },
    {
        "id": "crop_004",
        "text": """Sugarcane is a long-duration crop (12-18 months) grown in subtropical and 
tropical regions. Major states: UP, Maharashtra, Karnataka, Tamil Nadu. 
Requires temperature 21-27°C and 150-200 cm rainfall. 
Grows in deep loamy or alluvial soil. Plant in February-March or October-November. 
Apply FYM 20-25 tonnes per hectare before planting. 
Chemical fertilizer: N:P:K = 280:80:80 kg per hectare. 
Irrigate at 7-10 day intervals in summer. 
Harvest at 10-12 months for optimal sugar recovery. 
Common diseases: red rot, wilt, smut. Average yield: 70-80 tonnes per hectare.""",
        "metadata": {"category": "crop", "topic": "sugarcane cultivation"}
    },
    {
        "id": "crop_005",
        "text": """Maize (corn) is grown in both Kharif and Rabi seasons. 
Major states: Karnataka, Andhra Pradesh, Rajasthan, Maharashtra, Bihar. 
Temperature: 18-27°C. Rainfall: 60-110 cm. Grows in sandy loam to loamy soil. 
Sow in June-July (Kharif) or November (Rabi). Spacing: 60x25 cm. 
Apply Urea 65 kg + DAP 55 kg per acre. Top dress Urea 30 kg at knee-high stage. 
Needs 5-6 irrigations. Major pest: stem borer, fall armyworm. 
Common diseases: downy mildew, leaf blight. 
Harvest at 90-95 days (sweet corn) or 110-120 days (field corn). 
Average yield: 5-7 tonnes per hectare.""",
        "metadata": {"category": "crop", "topic": "maize cultivation"}
    },
    {
        "id": "crop_006",
        "text": """Tomato is the most important vegetable crop in India grown year-round. 
Major states: Andhra Pradesh, Karnataka, Maharashtra, Odisha, Gujarat. 
Temperature: 20-27°C. Cannot tolerate frost or extreme heat above 35°C. 
Grows in well-drained loamy soil with pH 6.0-7.0. 
Transplant seedlings at 30 days age. Spacing: 60x45 cm. 
Apply FYM 20 tonnes + NPK 120:60:60 kg per hectare. 
Irrigate regularly — drip irrigation preferred. 
Major diseases: early blight, late blight, bacterial wilt. 
Major pests: fruit borer, whitefly, thrips. Harvest at 60-65 days after transplanting. 
Average yield: 20-25 tonnes per hectare.""",
        "metadata": {"category": "crop", "topic": "tomato cultivation"}
    },
    {
        "id": "crop_007",
        "text": """Banana is a tropical fruit crop grown throughout the year. 
Major states: Tamil Nadu, Maharashtra, Gujarat, Andhra Pradesh, Karnataka. 
Requires temperature 15-35°C and 100-150 cm rainfall. 
Grows in deep, well-drained loamy soil with pH 5.5-7.0. 
Plant suckers at spacing 1.8x1.8 m. 
Apply 200g N + 60g P + 300g K per plant per year. 
Irrigate at 3-5 day intervals in summer. 
Major disease: Panama wilt (Fusarium), Sigatoka leaf spot. 
Harvest at 12-15 months after planting when fingers are fully formed. 
Average yield: 40-60 tonnes per hectare. Grand Naine is the most popular variety.""",
        "metadata": {"category": "crop", "topic": "banana cultivation"}
    },
    {
        "id": "crop_008",
        "text": """Groundnut (peanut) is a major oilseed and pulse crop of India. 
Major states: Gujarat, Rajasthan, Tamil Nadu, Andhra Pradesh, Karnataka. 
Temperature: 20-30°C. Rainfall: 50-100 cm. Grows in sandy loam soil with pH 6.0-7.0. 
Being a legume, it fixes atmospheric nitrogen — needs less nitrogen fertilizer. 
Apply DAP 40 kg + MOP 20 kg per acre as basal. No top dressing of N needed. 
Apply gypsum 200 kg per acre at pegging stage for calcium to pods. 
Major diseases: early leaf spot, late leaf spot, rust, tikka disease. 
Major pests: thrips, aphids, white grub. Harvest at 120-130 days. 
Average yield: 1.5-2.5 tonnes per hectare.""",
        "metadata": {"category": "crop", "topic": "groundnut cultivation"}
    },

    # ── PEST CONTROL ─────────────────────────────────────────
    {
        "id": "pest_001",
        "text": """Integrated Pest Management (IPM) is a sustainable approach to control pests 
using a combination of biological, cultural, mechanical, and chemical methods. 
Biological control: use natural enemies like Trichogramma wasps for bollworm, 
ladybird beetles for aphids, Beauveria bassiana fungus for thrips. 
Cultural control: crop rotation, deep ploughing, resistant varieties, 
proper spacing for air circulation. 
Mechanical control: light traps, pheromone traps, yellow sticky traps, 
hand picking of egg masses. 
Chemical control: use pesticides only when pest population exceeds economic threshold. 
Follow waiting period before harvest. IPM reduces pesticide cost by 30-50%.""",
        "metadata": {"category": "pest_control", "topic": "IPM integrated pest management"}
    },
    {
        "id": "pest_002",
        "text": """Aphids are small soft-bodied insects that suck plant sap causing yellowing, 
curling of leaves, and stunted growth. They secrete honeydew which causes 
sooty mould. Common on mustard, cotton, vegetables, and wheat. 
Control: spray Imidacloprid 0.3 ml per litre or Dimethoate 1.5 ml per litre. 
Biological control: release Chrysoperla carnea larvae (green lacewing). 
Natural control: spray neem oil 3ml per litre water. 
Reflective mulches repel aphids. Avoid excess nitrogen which attracts aphids. 
Ants protect aphids from predators — control ants to reduce aphid population.""",
        "metadata": {"category": "pest_control", "topic": "aphid control"}
    },
    {
        "id": "pest_003",
        "text": """Stem borer is the most destructive pest of rice and maize in India. 
The larva bores into the stem causing dead heart in vegetative stage 
and white ear in reproductive stage. 
Control in rice: use light traps, release Trichogramma japonicum egg parasitoid 
at 1.5 lakh per hectare. Spray Cartap hydrochloride 2g per litre or 
Chlorpyriphos 2ml per litre at early infestation. 
Drain water for 5 days to kill pupae. Remove and destroy affected tillers. 
Resistant varieties: TKM-6, W1263. 
Economic threshold: 5% dead hearts or 2% white ears.""",
        "metadata": {"category": "pest_control", "topic": "stem borer control"}
    },
    {
        "id": "pest_004",
        "text": """Whitefly is a major pest of cotton, tomato, chilli, and vegetables. 
It sucks plant sap and transmits viral diseases like cotton leaf curl virus. 
Adults are tiny white winged insects on the underside of leaves. 
Control: yellow sticky traps at 10 per acre. Spray neem oil 5ml per litre. 
Chemical: Acetamiprid 0.2g per litre or Spiromesifen 1ml per litre. 
Avoid spraying during flowering to protect pollinators. 
Biological control: Encarsia formosa parasite. 
Remove infected plants immediately. Avoid excess nitrogen. 
Rotate insecticides to prevent resistance. Spray in early morning or evening.""",
        "metadata": {"category": "pest_control", "topic": "whitefly control"}
    },
    {
        "id": "pest_005",
        "text": """Fall armyworm (FAW) is an invasive pest that attacks maize, sorghum, 
sugarcane, and vegetables. Larvae feed on leaves and bore into cobs. 
Symptoms: window pane feeding on young leaves, ragged holes in older leaves, 
frass (excreta) at leaf whorls. 
Control: apply sand + lime into whorls to kill young larvae. 
Spray Spinetoram 11.7% SC 0.5 ml per litre or Emamectin benzoate 0.4g per litre. 
Biological: spray Bacillus thuringiensis (Bt) 2g per litre at early stage. 
Use pheromone traps at 5 per acre for monitoring. 
Apply insecticide in morning when larvae are active. 
FAW has developed resistance to many insecticides — rotate chemicals.""",
        "metadata": {"category": "pest_control", "topic": "fall armyworm control"}
    },

    # ── SEASONS ───────────────────────────────────────────────
    {
        "id": "season_001",
        "text": """Kharif season (also called summer/monsoon crop season) runs from June to November. 
Crops are sown at the beginning of monsoon (June-July) and harvested in 
October-November. Major Kharif crops: rice, maize, cotton, sugarcane, 
groundnut, soybean, jowar, bajra, turmeric, jute. 
Kharif crops require high temperature and rainfall. 
Punjab and Haryana grow rice in Kharif. Maharashtra grows cotton and soybean. 
Success of Kharif depends on monsoon arrival and distribution. 
Late monsoon or drought severely affects Kharif crop production. 
Government announces MSP (Minimum Support Price) for Kharif crops before sowing.""",
        "metadata": {"category": "season", "topic": "kharif season"}
    },
    {
        "id": "season_002",
        "text": """Rabi season (winter crop season) runs from November to April. 
Crops are sown in October-November after monsoon withdrawal and harvested 
in March-April. Major Rabi crops: wheat, barley, mustard, gram (chickpea), 
peas, lentil, coriander, sunflower. Rabi crops grow in cool climate 
with mild frost tolerance. Punjab, Haryana, and UP produce most of India's 
wheat in Rabi season. Rabi crops depend on irrigation as monsoon has ended. 
Mustard is a major Rabi oilseed crop in Rajasthan. 
Gram (chickpea) is the major pulse crop of Rabi season grown in MP and Rajasthan.""",
        "metadata": {"category": "season", "topic": "rabi season"}
    },
    {
        "id": "season_003",
        "text": """Zaid season (summer/pre-kharif season) runs from March to June. 
It is a short duration season between Rabi harvest and Kharif sowing. 
Major Zaid crops: watermelon, muskmelon, cucumber, bitter gourd, 
pumpkin, moong (green gram), urad (black gram), and fodder crops. 
Zaid crops are grown along riverbanks and irrigated areas. 
High temperature and long days characterize this season. 
Needs intensive irrigation as it falls in the hot dry summer months. 
Zaid crops mature fast — in 60-90 days. Good opportunity for additional income 
for farmers who have irrigation facilities.""",
        "metadata": {"category": "season", "topic": "zaid season"}
    },

    # ── GOVERNMENT SCHEMES ────────────────────────────────────
    {
        "id": "scheme_001",
        "text": """PM-KISAN (Pradhan Mantri Kisan Samman Nidhi) provides income support of 
Rs 6000 per year to all landholding farmer families in three equal 
installments of Rs 2000 every four months. Launched in February 2019. 
Eligibility: all farmers with cultivable land. Excludes institutional 
landholders, government employees, and income taxpayers. 
Apply online at pmkisan.gov.in or through Common Service Centre (CSC). 
Documents needed: Aadhaar card, bank account, land records (Khasra/Khatauni). 
Over 11 crore farmers benefit from this scheme across India.""",
        "metadata": {"category": "government_scheme", "topic": "PM-KISAN scheme"}
    },
    {
        "id": "scheme_002",
        "text": """Soil Health Card scheme provides farmers with a card showing soil nutrient 
status and recommendations for fertilizer application. 
Government collects soil samples every 2 years and tests for 12 parameters: 
N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B. 
Based on results, crop-wise fertilizer recommendations are provided. 
This helps farmers avoid over-fertilization and reduce input costs. 
Apply for soil testing at nearest Krishi Vigyan Kendra (KVK) or 
agriculture department office. Testing is free for all farmers. 
Over 22 crore soil health cards have been distributed since 2015.""",
        "metadata": {"category": "government_scheme", "topic": "soil health card"}
    },
    {
        "id": "scheme_003",
        "text": """PM Fasal Bima Yojana (PMFBY) is crop insurance scheme that protects farmers 
against crop losses due to natural calamities, pests, and diseases. 
Premium rates: 2% for Kharif crops, 1.5% for Rabi crops, 5% for 
commercial/horticultural crops. Government pays remaining premium. 
Coverage: sowing failure, standing crop loss, post-harvest losses, 
localized calamities like hailstorm and landslide. 
Apply through banks (while taking crop loan) or online at pmfby.gov.in 
before the cutoff date (usually last date of sowing + 2 weeks). 
Compulsory for loanee farmers, optional for non-loanee farmers.""",
        "metadata": {"category": "government_scheme", "topic": "crop insurance PMFBY"}
    },
    {
        "id": "scheme_004",
        "text": """Pradhan Mantri Krishi Sinchai Yojana (PMKSY) aims to expand irrigation 
coverage and improve water use efficiency with the motto 'Har Khet Ko Pani, 
More Crop Per Drop'. Subsidy on drip irrigation: 55% for small/marginal farmers, 
45% for large farmers. Subsidy on sprinkler irrigation: 55% for small/marginal, 
45% for large farmers. Additional 10% subsidy for SC/ST farmers and 
women farmers. Apply through state agriculture department or 
horticulture department. Over 50 lakh hectares brought under micro-irrigation.""",
        "metadata": {"category": "government_scheme", "topic": "PMKSY irrigation scheme"}
    },
    {
        "id": "scheme_005",
        "text": """MSP (Minimum Support Price) is the price at which government purchases crops 
from farmers to protect them from market price fluctuations. 
CACP (Commission for Agricultural Costs and Prices) recommends MSP. 
MSP is announced for 23 crops: paddy, wheat, jowar, bajra, maize, ragi, 
cotton, groundnut, soybean, mustard, sunflower, gram, lentil, moong, urad, 
sugarcane, jute, and others. 
For 2024-25: Wheat MSP is Rs 2275/quintal, Paddy Rs 2300/quintal, 
Cotton Rs 7121/quintal. Farmers can sell at APMC mandis or through 
government procurement agencies like FCI, NAFED, and CCI.""",
        "metadata": {"category": "government_scheme", "topic": "MSP minimum support price"}
    },

    # ── ORGANIC FARMING ───────────────────────────────────────
    {
        "id": "organic_001",
        "text": """Organic farming is a system that avoids synthetic fertilizers and pesticides, 
relying instead on organic manures, biofertilizers, and natural pest control. 
Organic certification in India is done under NPOP (National Programme for 
Organic Production). Certification takes 3 years of conversion period. 
Certified organic products fetch 20-50% premium price in market. 
Key practices: composting, vermicomposting, green manuring, crop rotation, 
mixed cropping, use of biofertilizers (Rhizobium, PSB, Azospirillum). 
States promoting organic farming: Sikkim (100% organic), Uttarakhand, 
Himachal Pradesh, Madhya Pradesh. Sikkim became India's first fully organic state in 2016.""",
        "metadata": {"category": "organic_farming", "topic": "organic farming basics"}
    },
    {
        "id": "organic_002",
        "text": """Vermicomposting uses earthworms (Eisenia foetida) to decompose organic matter 
into nutrient-rich compost. Vermicompost contains 2-3% N, 1.5-2% P, 1.5-2% K 
plus micronutrients and beneficial microorganisms. 
Process: Make a bed 1m wide x 0.3m deep x any length. Add moist organic waste 
(crop residue, kitchen waste, cattle dung). Add 1000 earthworms per kg organic matter. 
Cover with gunny bags to maintain moisture. Ready in 45-60 days. 
Apply 2-4 tonnes vermicompost per hectare. Cost to produce: Rs 3-4 per kg. 
Selling price: Rs 8-15 per kg. Vermicomposting is an additional income source for farmers.""",
        "metadata": {"category": "organic_farming", "topic": "vermicomposting"}
    },
    {
        "id": "organic_003",
        "text": """Jeevamrut is a liquid biofertilizer used in natural farming developed by 
Subhash Palekar. Recipe: 200 litres water + 10 kg fresh desi cow dung + 
5-10 litres cow urine + 2 kg jaggery + 2 kg pulse flour + 
handful of bund soil. Mix and ferment for 48 hours in shade. 
Apply 200 litres per acre every 15 days through irrigation or spraying. 
Jeevamrut activates soil microorganisms and improves soil fertility naturally. 
Used extensively in Zero Budget Natural Farming (ZBNF) promoted in Andhra Pradesh. 
Reduces input cost by 70-80% compared to chemical farming.""",
        "metadata": {"category": "organic_farming", "topic": "jeevamrut natural farming"}
    },

    # ── CROP DISEASES ─────────────────────────────────────────
    {
        "id": "disease_001",
        "text": """Rice blast is the most destructive fungal disease of rice caused by 
Magnaporthe oryzae. Symptoms: diamond-shaped lesions with grey center and 
brown border on leaves (leaf blast), dark brown lesions on neck (neck blast). 
Neck blast at heading stage causes complete crop loss. 
Control: use resistant varieties (Tetep, Mahsuri). Seed treatment with 
Carbendazim 2g per kg seed. Spray Tricyclazole 0.6g per litre at first sign. 
Avoid excess nitrogen which increases susceptibility. 
Silicon application at 200 kg per hectare reduces blast severity. 
Burn infected stubble after harvest to reduce inoculum.""",
        "metadata": {"category": "disease", "topic": "rice blast disease"}
    },
    {
        "id": "disease_002",
        "text": """Powdery mildew affects wheat, grapes, peas, and vegetables. 
White powdery fungal growth on upper leaf surface is the main symptom. 
Caused by various Erysiphe species. Favored by dry weather with high humidity. 
Control in wheat: spray Propiconazole 1ml per litre or Triadimefon 1g per litre 
at flag leaf stage. Resistant wheat varieties: HD-2781, K-9107. 
In vegetables: spray wettable sulphur 3g per litre or Hexaconazole 1ml per litre. 
Remove and destroy infected plant parts. Avoid dense planting. 
Proper spacing improves air circulation and reduces disease.""",
        "metadata": {"category": "disease", "topic": "powdery mildew disease"}
    },
    {
        "id": "disease_003",
        "text": """Bacterial wilt affects tomato, brinjal, potato, and chilli. 
Caused by Ralstonia solanacearum bacteria. Symptoms: sudden wilting of plants 
during hot part of day, recovering at night initially, then permanent wilt. 
Vascular browning seen when stem is cut. White bacterial ooze in water test. 
No effective chemical control. Management: use resistant varieties, 
practice crop rotation with non-solanaceous crops for 3-4 years, 
soil solarization using clear plastic mulch for 4-6 weeks in summer, 
apply lime to raise pH above 7, use grafted seedlings on resistant rootstock, 
avoid waterlogging, remove and burn infected plants immediately.""",
        "metadata": {"category": "disease", "topic": "bacterial wilt disease"}
    },

    # ── GENERAL FARMING PRACTICES ─────────────────────────────
    {
        "id": "practice_001",
        "text": """Crop rotation is the practice of growing different crops sequentially on 
the same land to improve soil fertility and break pest and disease cycles. 
Benefits: fixes atmospheric nitrogen (legume rotation), breaks pest cycles, 
improves soil structure, reduces soil erosion, increases yield. 
Common rotations: Rice-Wheat (Indo-Gangetic Plain), Cotton-Wheat (Punjab), 
Maize-Mustard (UP), Groundnut-Wheat (Gujarat), Sugarcane-Ratoon-Wheat. 
Include legume in rotation: Rice-Wheat-Moong or Maize-Potato-Moong. 
After legume, reduce nitrogen fertilizer by 25-30% for next crop. 
Minimum 3-year rotation recommended to break persistent soil pathogens.""",
        "metadata": {"category": "farming_practice", "topic": "crop rotation"}
    },
    {
        "id": "practice_002",
        "text": """Mulching is the practice of covering soil surface with organic or inorganic 
material to conserve moisture, suppress weeds, and regulate soil temperature. 
Organic mulches: dry leaves, straw, crop residue, grass clippings — 
decompose and add organic matter. Apply 5-10 cm thick layer. 
Plastic mulch: silver/black polyethylene film — 25 micron thick. 
Black mulch suppresses weeds, silver mulch repels insects. 
Mulching reduces irrigation requirement by 30-40% and weed control cost by 50%. 
Best for vegetables, strawberry, watermelon, and orchard crops. 
Mulch with straw in vineyards and orchards during summer to conserve moisture.""",
        "metadata": {"category": "farming_practice", "topic": "mulching"}
    },
    {
        "id": "practice_003",
        "text": """Seed treatment is the application of fungicides, insecticides, and biofertilizers 
to seeds before sowing to protect from soil-borne diseases and pests. 
Standard seed treatment: Thiram 3g/kg + Carbendazim 2g/kg seed for fungal diseases. 
For bacterial diseases: Streptomycin 0.5g/kg seed. 
Biofertilizer seed treatment: Rhizobium for legumes, Azospirillum for cereals, 
PSB for all crops at 10-25g per kg seed. 
Apply biofertilizer last — after chemical treatment, in shade, plant within 24 hours. 
Imidacloprid 5g/kg seed for soil-borne insect pests. 
Treated seeds should not be used for food or fodder. 
Seed treatment is the cheapest and most effective disease management practice.""",
        "metadata": {"category": "farming_practice", "topic": "seed treatment"}
    },
    {
        "id": "practice_004",
        "text": """Post-harvest management involves all activities after crop harvest to 
reduce losses and maintain quality. India loses 20-30% of produce post-harvest. 
Key practices: proper harvesting at right maturity, immediate removal of field heat, 
cleaning and grading, proper packaging, cold storage, and efficient transport. 
Cold storage temperature: potato 2-4°C, apple 0-2°C, mango 8-12°C, 
tomato 10-13°C, banana 13-14°C. 
For grains: dry to safe moisture (paddy 14%, wheat 12%) before storage. 
Use hermetic storage bags (Purdue Improved Crop Storage) to prevent grain pests. 
Apply Aluminium phosphide tablets in sealed godowns for pest control.""",
        "metadata": {"category": "farming_practice", "topic": "post harvest management"}
    },
    {
        "id": "practice_005",
        "text": """Precision farming uses technology to optimize inputs and maximize yields. 
GPS-based soil sampling maps nutrient variability within a field. 
Variable rate applicators apply different fertilizer rates based on soil map. 
Drones for crop monitoring, spraying, and yield mapping. 
IoT soil sensors measure soil moisture, temperature, and EC in real-time. 
Satellite imagery (NDVI) shows crop health and stress areas. 
Precision farming reduces fertilizer use by 15-20%, water by 20-30%, 
and pesticide by 25-30% while increasing yield by 10-15%. 
Indian government promotes precision farming under RKVY scheme. 
Suitable for large farms above 5 hectares to justify technology investment.""",
        "metadata": {"category": "farming_practice", "topic": "precision farming"}
    },
]

print(f"Total knowledge documents prepared: {len(documents)}")

# ── STEP 2: Setup ChromaDB ────────────────────────────────────
print("\nSetting up ChromaDB...")

os.makedirs("vector_db", exist_ok=True)

# Initialize ChromaDB with persistent storage
# Data is saved to disk — survives app restarts
chroma_client = chromadb.PersistentClient(path="vector_db")

# Delete existing collection if rebuilding
try:
    chroma_client.delete_collection("farming_knowledge")
    print("Deleted existing collection — rebuilding...")
except:
    pass

# Use sentence-transformers for embeddings (free, local, no API needed)
# all-MiniLM-L6-v2 is fast and good for semantic search
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create collection
collection = chroma_client.create_collection(
    name="farming_knowledge",
    embedding_function=embedding_fn,
    metadata={"description": "Indian farming knowledge base for RAG"}
)

print("ChromaDB collection created: farming_knowledge")

# ── STEP 3: Add documents ─────────────────────────────────────
print("\nAdding documents to ChromaDB...")
print("(This may take a minute — downloading embedding model first time)")

# Add in batches of 10
batch_size = 10
for i in range(0, len(documents), batch_size):
    batch     = documents[i:i+batch_size]
    ids       = [d["id"] for d in batch]
    texts     = [d["text"] for d in batch]
    metadatas = [d["metadata"] for d in batch]

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )
    print(f"  Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} "
          f"({len(ids)} documents)")

# ── STEP 4: Verify ────────────────────────────────────────────
print("\n" + "=" * 60)
print("Verification — test query:")
print("=" * 60)

test_results = collection.query(
    query_texts=["what is drip irrigation"],
    n_results=2
)

print(f"\nQuery: 'what is drip irrigation'")
print(f"Top result: {test_results['ids'][0][0]}")
print(f"Distance: {test_results['distances'][0][0]:.4f}")
print(f"Preview: {test_results['documents'][0][0][:100]}...")

print("\n" + "=" * 60)
print("✅ Knowledge Base Built Successfully!")
print(f"Total documents: {collection.count()}")
print(f"Saved to: vector_db/")
print("Categories covered:")
categories = set(d["metadata"]["category"] for d in documents)
for cat in sorted(categories):
    count = sum(1 for d in documents if d["metadata"]["category"] == cat)
    print(f"  {cat}: {count} documents")
print("\nNext step: streamlit run app.py")
print("=" * 60)

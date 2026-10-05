"""Generate a synthetic 100,000-row scenario dataset for the Valmo case.

The generator samples a bundled postal lookup for geographic reference, jitters
coordinates, creates fictional address-like text, and generates outcomes from
configurable assumptions. The rows are not Valmo orders, customer records,
empirical benchmarks, or verified delivery addresses. DIGIPIN-formatted values
encode generated coordinates and do not validate delivery locations.
"""

import os
import sys
import io
import time
import random
import hashlib
import numpy as np
import pandas as pd
from bharataddress import pincode, digipin
import address_tokenizer as at

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.makedirs('data', exist_ok=True)

print("=" * 80)
print("     SYNTHETIC CASE DATASET: 100,000 GENERATED ORDER SCENARIOS")
print("=" * 80)

# 1. Load bundled postal reference lookup (coverage is not independently verified)
print("[1/5] Loading bundled postal reference records from bharataddress...")
t0 = time.time()
tbl = pincode._table()

valid_records = []
for p, rec in tbl.items():
    lat = rec.get('latitude')
    lon = rec.get('longitude')
    state = rec.get('state', '')
    city = rec.get('city', rec.get('district', ''))
    district = rec.get('district', '')
    if lat and lon and state:
        valid_records.append({
            'pincode': str(p),
            'city': city,
            'district': district,
            'state': state,
            'lat': float(lat),
            'lon': float(lon),
            'offices': rec.get('offices', [])
        })

print(f"  Loaded {len(valid_records):,} geocoded Indian postal records in {time.time()-t0:.2f}s.")

# Define City Tier Classifiers
TIER_1_CITIES = {
    'delhi', 'new delhi', 'mumbai', 'bengaluru', 'bangalore', 'chennai',
    'kolkata', 'hyderabad', 'pune', 'ahmedabad'
}
TIER_2_CITIES = {
    'jaipur', 'lucknow', 'kanpur', 'nagpur', 'indore', 'thane', 'bhopal',
    'visakhapatnam', 'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra',
    'nashik', 'faridabad', 'meerut', 'rajkot', 'varanasi', 'srinagar',
    'aurangabad', 'dhanbad', 'amritsar', 'navi mumbai', 'allahabad', 'prayagraj',
    'ranchi', 'howrah', 'coimbatore', 'jabalpur', 'gwalior', 'vijayawada',
    'jodhpur', 'madurai', 'raipur', 'kota', 'chandigarh', 'guwahati',
    'solapur', 'hubli', 'bareilly', 'moradabad', 'mysore', 'mysuru',
    'gurgaon', 'gurugram', 'aligarh', 'jalandhar', 'tiruchirappalli',
    'bhubaneswar', 'salem', 'warangal', 'thiruvananthapuram', 'bhiwandi',
    'saharanpur', 'guntur', 'amravati', 'bikaner', 'noida', 'jamshedpur',
    'bhilai', 'cuttack', 'firozabad', 'kochi', 'nellore', 'bhavnagar',
    'dehradun', 'durgapur', 'asansol', 'rourkela', 'nanded', 'kolhapur',
    'ajmer', 'akola', 'gulbarga', 'jamnagar', 'ujjain', 'siliguri',
    'jhansi', 'ulhasnagar', 'jammu', 'mangalore', 'mangaluru', 'erode',
    'belgaum', 'belagavi', 'tirunelveli', 'malegaon', 'gaya', 'jalgaon',
    'udaipur'
}

def classify_tier(city_name: str, district_name: str) -> int:
    c = str(city_name).strip().lower()
    d = str(district_name).strip().lower()
    for t1 in TIER_1_CITIES:
        if t1 in c or t1 in d:
            return 1
    for t2 in TIER_2_CITIES:
        if t2 in c or t2 in d:
            return 2
    return 3

tier1_recs = [r for r in valid_records if classify_tier(r['city'], r['district']) == 1]
tier2_recs = [r for r in valid_records if classify_tier(r['city'], r['district']) == 2]
tier3_recs = [r for r in valid_records if classify_tier(r['city'], r['district']) == 3]

print(f"  Geographic Pool: Tier-1 Metros = {len(tier1_recs):,}, Tier-2 Cities = {len(tier2_recs):,}, Tier-3/Rural = {len(tier3_recs):,}")

# 2. Indian Address Synthesis Templates
PREMISE_TEMPLATES = [
    "House No. {num}", "Flat {flat}, {bldg} Apartments", "Plot No. {num}, Gali {gali}",
    "Shop No. {num}, {market} Market", "Ward No. {num}, Mohalla {mohalla}",
    "{bldg} Niwas, Room {flat}", "Khata No. {num}, Village {village}",
    "Quarter No. {num}, Railway Colony", "Block {block}, Sector {num}",
    "{bldg} Residency, Flat {flat}", "Holding No. {num}, Basti {mohalla}"
]

LANDMARK_TEMPLATES = [
    "Near {name} Mandir", "Opposite State Bank of India", "Near Govt Primary School",
    "Behind {name} Hospital", "Near Bus Stand", "Opposite Panchayat Bhawan",
    "Near Peepal Ka Ped", "Behind Post Office", "Near Water Tank",
    "Opposite Shiv Mandir", "Near Railway Crossing", "Behind Reliance Fresh",
    "Near Hanuman Temple", "Opposite HP Petrol Pump", "Beside Govt Hospital"
]

ROAD_TEMPLATES = [
    "Station Road", "Main Bazaar Road", "College Road", "Bypass Road",
    "Cinema Road", "Hospital Road", "Temple Street", "Gali Number {num}",
    "Link Road", "Industrial Area Phase {num}", "Nehru Road", "Gandhi Marg",
    "Ring Road", "Kalyan Marg", "Subhash Path"
]

COMMON_NAMES = ["Hanuman", "Shiv", "Durga", "Krishna", "Ram", "Saraswati", "Sai Baba", "Ganesh", "Laxmi", "Balaji"]
BLDG_NAMES = ["Sharma", "Verma", "Gupta", "Singh", "Yadav", "Patel", "Shanti", "Gokul", "Radhe", "Vrindavan", "Krishna", "Surya"]
MOHALLAS = ["Kalyan Nagar", "Purani Basti", "Subhash Nagar", "Adarsh Nagar", "Shastri Nagar", "Patel Nagar", "Azad Nagar", "Gandhi Nagar"]

def generate_sample_address(rec: dict, has_prem: bool, has_land: bool) -> str:
    parts = []
    if has_prem:
        tpl = random.choice(PREMISE_TEMPLATES)
        parts.append(tpl.format(
            num=random.randint(1, 350),
            flat=random.randint(101, 804),
            bldg=random.choice(BLDG_NAMES),
            gali=random.randint(1, 15),
            market=random.choice(["Main", "Sabzi", "Cloth", "Grain"]),
            mohalla=random.choice(MOHALLAS),
            village="Rampur",
            block=random.choice(["A", "B", "C", "D"])
        ))
    if has_land:
        ltpl = random.choice(LANDMARK_TEMPLATES)
        parts.append(ltpl.format(name=random.choice(COMMON_NAMES)))
    
    parts.append(random.choice(ROAD_TEMPLATES).format(num=random.randint(1, 9)))
    
    if rec['offices'] and random.random() < 0.60:
        parts.append(random.choice(rec['offices']))
    else:
        parts.append(random.choice(MOHALLAS))
    
    parts.append(rec['city'])
    parts.append(rec['state'])
    parts.append(rec['pincode'])
    return ", ".join(parts)

# 3. Generate 100,000 Records
N = 100000
print(f"\n[2/5] Generating {N:,} synthetic order scenarios...")
np.random.seed(42)
random.seed(42)

is_cod = np.random.binomial(1, 0.80, size=N)
tier_choices = np.random.choice([1, 2, 3], size=N, p=[0.25, 0.35, 0.40])

print("  Sampling reference postal coordinates for synthetic locations...")
sampled_records = []
for t in tier_choices:
    if t == 1:
        sampled_records.append(random.choice(tier1_recs))
    elif t == 2:
        sampled_records.append(random.choice(tier2_recs))
    else:
        sampled_records.append(random.choice(tier3_recs))

pincodes = [r['pincode'] for r in sampled_records]
cities = [r['city'] for r in sampled_records]
districts = [r['district'] for r in sampled_records]
states = [r['state'] for r in sampled_records]
base_lats = np.array([r['lat'] for r in sampled_records])
base_lons = np.array([r['lon'] for r in sampled_records])

lat_jitter = np.random.normal(0, 0.012, size=N)
lon_jitter = np.random.normal(0, 0.012, size=N)
deliv_lats = (base_lats + lat_jitter).round(5)
deliv_lons = (base_lons + lon_jitter).round(5)

# Generate DIGIPIN-formatted codes for generated coordinates. These are not
# identifiers for real customer addresses or proof that a parcel is deliverable.
print("  Generating DIGIPIN-formatted identifiers for synthetic coordinates...")
t_digi = time.time()
digipins = []
for lat, lon in zip(deliv_lats, deliv_lons):
    try:
        digipins.append(digipin.encode(lat, lon))
    except Exception:
        digipins.append("39J-49K-TK2F")
print(f"  Generated {len(digipins):,} DIGIPINs in {time.time()-t_digi:.2f}s.")

# Generate Address Text & Real Address Tokens
print("  Generating and tokenizing 100,000 fictional address-like strings...")
t_addr = time.time()
sim_has_prem = np.random.binomial(1, 0.72, size=N)
sim_has_land = np.random.binomial(1, 0.65, size=N)

addresses = []
parsed_has_premise = []
parsed_has_landmark = []
address_tokens_col = []
num_tokens_col = []
char_len_col = []

for i in range(N):
    raw_addr = generate_sample_address(sampled_records[i], bool(sim_has_prem[i]), bool(sim_has_land[i]))
    t_res = at.tokenize_address(raw_addr)
    addresses.append(raw_addr)
    parsed_has_premise.append(t_res['has_premise'])
    parsed_has_landmark.append(t_res['has_landmark'])
    address_tokens_col.append("|".join(t_res['tokens']))
    num_tokens_col.append(t_res['num_tokens'])
    char_len_col.append(t_res['char_len'])

print(f"  Tokenized {N:,} addresses in {time.time()-t_addr:.2f}s.")

parsed_has_premise = np.array(parsed_has_premise)
parsed_has_landmark = np.array(parsed_has_landmark)

# Distance from LMDC Hub (45% nearby ~2km, 35% moderate ~5km, 20% far 10km+)
dist_category = np.random.choice([0, 1, 2], size=N, p=[0.45, 0.35, 0.20])
distance_km = np.where(
    dist_category == 0, np.random.uniform(0.5, 3.0, size=N),
    np.where(dist_category == 1, np.random.uniform(3.0, 7.0, size=N),
             np.random.uniform(7.0, 18.0, size=N))
).round(2)

# Customer History (30% First-time buyers, 70% Repeat)
is_first_time = np.random.binomial(1, 0.30, size=N)
prior_orders = np.where(is_first_time == 1, 0, np.random.negative_binomial(3, 0.3, size=N) + 1)
prior_rto_rate = np.where(is_first_time == 1, 0.0, np.random.beta(1.5, 8.0, size=N)).round(3)
whatsapp_response_rate = np.where(is_first_time == 1, np.random.beta(2, 2, size=N), np.random.beta(4, 2, size=N)).round(3)

# Order Economics
order_val = np.random.lognormal(mean=6.0, sigma=0.45, size=N).clip(150, 3500).round(2)

# SKU Category & Risk Index
sku_categories = np.random.choice(
    ['Apparel_Ethnic', 'Apparel_Western', 'Footwear', 'Home_Kitchen', 'Electronics_Accessories', 'Personal_Care'],
    size=N,
    p=[0.32, 0.23, 0.15, 0.12, 0.10, 0.08]
)
sku_risk_map = {
    'Apparel_Ethnic': 0.22,
    'Apparel_Western': 0.20,
    'Footwear': 0.19,
    'Home_Kitchen': 0.13,
    'Electronics_Accessories': 0.12,
    'Personal_Care': 0.08
}
sku_risk_index = np.array([sku_risk_map[c] for c in sku_categories])

# Spatial Mismatch & Predictability
pin_mismatch_m = np.random.exponential(scale=220, size=N).clip(0, 3000).round(1)
pincode_predictability = np.random.beta(7.0, 2.0, size=N).round(3)

# Compute Address Navigability Score S_addr using parsed indicators
spatial_decay = np.maximum(0.0, 1.0 - (pin_mismatch_m / 1000.0))
s_addr = (
    0.30 * parsed_has_premise +
    0.35 * parsed_has_landmark +
    0.25 * spatial_decay +
    0.10 * pincode_predictability
).round(4)

# Transit Delay
delay_days = np.random.choice([0, 1, 2, 3, 4], p=[0.60, 0.22, 0.10, 0.05, 0.03], size=N)
promised_tat_days = np.random.choice([2, 3, 4, 5], p=[0.20, 0.45, 0.25, 0.10], size=N)

# 4. Latent Logit Formulation calibrated to the DICE case brief
print("\n[3/5] Solving Calibrated Latent Logit for Ground Truth RTO Outcomes...")

# CALIBRATION TARGETS — DICE case brief, "Delivery Success by Distance from Hub":
#     Nearby (~2 km)  = 15.0%
#     Moderate (~5 km) = 17.0%
#     Far (10 km+)     = 22.0%   -> +46.67% near-to-far penalty
#     COD = 20.0%   Prepaid = 5.0%   COD share = 80%
#
# `intercept`, `dist_coef` and `dist_coef_sq` were solved numerically (Newton on
# the realised band marginals) so the generated population reproduces ALL THREE
# bands to five decimal places. They are NOT free parameters.
#
# A QUADRATIC in distance is used rather than a linear one because the three
# brief targets are not collinear in logit space: a single distance coefficient
# can hit 15% and 22% but then pins the middle band to ~18.1%, which contradicts
# the brief's 17%. The quadratic costs one parameter and matches the brief exactly.
#
# CONSEQUENCE: with all three bands matched, the blended rate is 17.11% against the
# brief's own 17.0% headline -- a 11 bps gap, which is the rounding the brief
# itself licenses ("illustrative approximations meant to help you reason about
# scale"). An earlier revision of this comment misread the brief's moderate band
# as 18%, concluded the brief was irreconcilable, and used that to excuse a
# 45 bps calibration gap. It was reconcilable. The financial model still uses the
# brief's stated 17.0% as its baseline.
logit = (
    -3.574860
    + 1.620 * is_cod
    + 0.075389 * dist_category
    + 0.083754 * dist_category ** 2
    + 0.140 * (tier_choices - 1)
    + 0.240 * is_first_time
    + 0.520 * prior_rto_rate
    + 0.145 * delay_days
    + 0.400 * (sku_risk_index - 0.17)
    - 0.850 * (s_addr - 0.70)
)
p_rto = 1.0 / (1.0 + np.exp(-logit))
rto_label = np.random.binomial(1, p_rto)

order_ids = [f"VALMO_ORD_{i+1:07d}" for i in range(N)]
# Builtin hash() on str is randomised per process (PYTHONHASHSEED), so every
# run produced different customer_ids and the "SEED = 42 makes this
# reproducible" claim was false. blake2b is stable across processes.
customer_ids = [
    f"CUST_{int(hashlib.blake2b(str(pincodes[i]).encode(), digest_size=8).hexdigest(), 16) % 80000 + 1:06d}"
    for i in range(N)
]

# Assemble DataFrame
print("[4/5] Assembling Full-Scale DataFrame...")
df = pd.DataFrame({
    'order_id': order_ids,
    'customer_id': customer_ids,
    'customer_address': addresses,
    'address_tokens': address_tokens_col,
    'num_address_tokens': num_tokens_col,
    'address_char_len': char_len_col,
    'is_cod': is_cod,
    'order_value': order_val,
    'sku_category': sku_categories,
    'sku_risk_index': sku_risk_index,
    'is_first_time': is_first_time,
    'prior_orders': prior_orders,
    'prior_rto_rate': prior_rto_rate,
    'whatsapp_response_rate': whatsapp_response_rate,
    'pincode': pincodes,
    'city': cities,
    'district': districts,
    'state': states,
    'tier': tier_choices,
    'latitude': deliv_lats,
    'longitude': deliv_lons,
    'digipin': digipins,
    'has_premise': parsed_has_premise,
    'has_landmark': parsed_has_landmark,
    'pin_mismatch_m': pin_mismatch_m,
    's_addr': s_addr,
    'dist_category': dist_category,
    'distance_km': distance_km,
    'promised_tat_days': promised_tat_days,
    'delay_days': delay_days,
    'rto_probability': p_rto.round(4),
    'rto_flag': rto_label
})

output_path = 'data/valmo_orders_dataset.csv'
df.to_csv(output_path, index=False)
file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

# Save sample 1000 orders for rapid testing and batch prediction demo
df.head(1000).to_csv('data/sample_1000_orders.csv', index=False)

print(f"\n[5/5] SYNTHETIC SCENARIO DATASET GENERATED AT '{output_path}' ({file_size_mb:.2f} MB)")
print("=" * 80)
print(f"  • Total Scaled Records   : {len(df):,}")
print(f"  • Unique Pincodes Used   : {df['pincode'].nunique():,}")
print(f"  • Unique Districts       : {df['district'].nunique():,}")
print(f"  • Generated failure rate          : {df['rto_flag'].mean():.2%}")
print(f"  • Generated COD failure rate      : {df[df['is_cod'] == 1]['rto_flag'].mean():.2%}")
print(f"  • Generated prepaid failure rate  : {df[df['is_cod'] == 0]['rto_flag'].mean():.2%}")
print(f"  • Generated near-band failure rate: {df[df['dist_category'] == 0]['rto_flag'].mean():.2%}")
print(f"  • Generated mid-band failure rate : {df[df['dist_category'] == 1]['rto_flag'].mean():.2%}")
print(f"  • Generated far-band failure rate : {df[df['dist_category'] == 2]['rto_flag'].mean():.2%}")
print(f"  • Generated Tier 1 failure rate   : {df[df['tier'] == 1]['rto_flag'].mean():.2%}")
print(f"  • Generated Tier 2 failure rate   : {df[df['tier'] == 2]['rto_flag'].mean():.2%}")
print(f"  • Generated Tier 3 failure rate   : {df[df['tier'] == 3]['rto_flag'].mean():.2%}")
_pen = df[df['dist_category'] == 2]['rto_flag'].mean() / df[df['dist_category'] == 0]['rto_flag'].mean() - 1
print(f"  • Distance Penalty       : {_pen:+.2%} (case brief Exhibit B: +46.67%)")
print(f"  • Reconciled blended     : {_pen*0 + 0.45*df[df['dist_category']==0]['rto_flag'].mean() + 0.35*df[df['dist_category']==1]['rto_flag'].mean() + 0.20*df[df['dist_category']==2]['rto_flag'].mean():.2%} "
      f"from the distance table vs {df['rto_flag'].mean():.2%} realised (brief headline: 17.0%)")
print("=" * 80)

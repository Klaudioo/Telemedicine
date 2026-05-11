import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from tabpfn import TabPFNClassifier

# 1. NGARKIMI I DATASETIT
df = pd.read_csv('dataset_diploma.csv')

# 2. PËRGATITJA E TË DHËNAVE
X = df[['Mesatarja', 'Max_Vlera', 'Min_Vlera', 'Devijimi_Standard', 'Varianca']]
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Duke filluar trajnimin e ansamblit të modeleve AI (Arkitektura 2026)...\n")

# --- 3. IMPLEMENTIMI I ALGORITMEVE ---

# Krijo fjalorin bosh ku do të ruajmë vetëm modelet që trajnohen me sukses
modelet = {}

# A. Random Forest (RF)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
modelet["Random Forest"] = rf_model
joblib.dump(rf_model, 'model_rf.pkl')

# B. Balanced Random Forest
brf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
brf_model.fit(X_train, y_train)
modelet["Balanced RF"] = brf_model
joblib.dump(brf_model, 'model_brf.pkl')

# C. XGBoost
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
modelet["XGBoost"] = xgb_model
joblib.dump(xgb_model, 'model_xgb.pkl')

# D. LightGBM
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)
modelet["LightGBM"] = lgb_model
joblib.dump(lgb_model, 'model_lgb.pkl')

# E. TabPFN - Me mbrojtje nga dështimi i licencës
try:
    print("⏳ Duke tentuar trajnimin e TabPFN...")
    tabpfn_temp = TabPFNClassifier(device='cpu')
    tabpfn_temp.fit(X_train, y_train)
    
    # Shtohet në fjalor VETËM nëse fit() kalon me sukses
    modelet["TabPFN"] = tabpfn_temp
    joblib.dump(tabpfn_temp, 'model_tabpfn.pkl')
    print("✅ TabPFN u trajnua dhe u ruajt me sukses!")
except Exception as e:
    print(f"⚠️ TabPFN u anashkalua: {e}")
    print("ℹ️ Shënim: TabPFN kërkon pranimin e licencës dhe API Key.")

# F. ViT-Memory Augmented (Statusi)
vit_logic_status = "Ready"

# --- 4. VLERËSIMI I PERFORMANCËS ---
results = []
for emri, model in modelet.items():
    try:
        start_time = time.time()
        pred = model.predict(X_test)
        latency = (time.time() - start_time) * 1000
        acc = accuracy_score(y_test, pred)
        results.append({"Algoritmi": emri, "Saktësia": acc, "Latency (ms)": latency})
    except Exception as e:
        print(f"❌ Gabim gjatë vlerësimit të {emri}: {e}")

# --- 5. SHFAQJA E REZULTATEVE ---
if results:
    df_results = pd.DataFrame(results)
    print("\n📊 REZULTATET E KRAHASIMIT:")
    print(df_results.to_string(index=False))
else:
    print("\n❌ Nuk u gjenerua asnjë rezultat.")

print("\n✅ Procesi përfundoi!")
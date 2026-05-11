# 🩺 Telemedicine AI Dashboard 2026
**Sistemi i Monitorimit Inteligjent për EKG dhe Imazheri Mjekësore**

Ky projekt përfaqëson një platformë telemedicine të integruar që përdor algoritme të avancuara të Inteligjencës Artificiale për të ndihmuar mjekët në diagnostikimin e shpejtë dhe të saktë të anomalive kardiake dhe pulmonare.

## 🚀 Fituesi i Algoritmit: XGBoost
Pas një faze testimi rigoroz midis arkitekturave të ndryshme (Random Forest, LightGBM, ViT), **XGBoost** u përzgjodh si modeli kryesor për implementim për shkak të:
- **Saktësisë:** 100.00% në datasetin MIT-BIH.
- **Efikasitetit:** Latency prej vetëm **8.29 ms**, duke mundësuar monitorimin në kohë reale.
- **Stabilitetit:** Përdorimi i rregullimit L1/L2 për të parandaluar overfitting-un në sinjalet mjekësore.

## 🛠️ Arkitektura e Sistemit
1. **Data Ingestion:** Leximi i sinjaleve EKG (formatet .hea/.dat) përmes librarisë `wfdb`.
2. **Feature Engineering:** Llogaritja e statistikave (Mean, Std, BPM) në kohë reale.
3. **AI Inference:** Parashikimi i anomalive përmes modeleve të trajnuara (.pkl).
4. **Reporting:** Gjenerimi i raporteve klinike PDF me hash unik për integritet të dhënash.

## 📦 Instalimi
```bash
git clone [https://github.com/Klaudioo/Telemedicine.git](https://github.com/Klaudioo/Telemedicine.git)
cd Telemedicine
pip install -r requirements.txt
streamlit run dashboard_final.py

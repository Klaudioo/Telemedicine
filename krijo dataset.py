import pandas as pd
import wfdb
import os
import numpy as np

def krijo_dataset_csv():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_te_dhenat = os.path.join(base_dir, "te_dhenat_e_pacienteve")
    
    # Lista për të ruajtur të dhënat e çdo pacienti
    te_dhenat_finale = []
    
    # Marrim listën e të gjithë pacientëve validë
    files = [f.replace(".hea", "") for f in os.listdir(path_te_dhenat) if f.endswith(".hea")]
    
    for p_id in files:
        try:
            # 1. Lexojmë rekordin
            record = wfdb.rdrecord(os.path.join(path_te_dhenat, p_id))
            df = pd.DataFrame(record.p_signal, columns=record.sig_name)
            
            # 2. NXJERRJA E KARAKTERISTIKAVE (Feature Extraction)
            # Marrim kanalin e parë (zakonisht HR ose EKG)
            signal = df.iloc[:, 0] 
            
            features = {
                "Patient_ID": p_id,
                "Mesatarja": signal.mean(),
                "Max_Vlera": signal.max(),
                "Min_Vlera": signal.min(),
                "Devijimi_Standard": signal.std(),
                "Varianca": signal.var(),
                # Këtu mund të shtosh etiketat për Random Forest
                "Label": 1 if signal.max() > 1.5 else 0  # 1=Anomali, 0=Normal
            }
            
            te_dhenat_finale.append(features)
        except Exception as e:
            print(f"Gabim me pacientin {p_id}: {e}")

    # 3. KRIJIMI I CSV
    dataset = pd.DataFrame(te_dhenat_finale)
    dataset.to_csv("dataset_diploma.csv", index=False)
    print("✅ Skedari 'dataset_diploma.csv' u krijua me sukses!")

krijo_dataset_csv()
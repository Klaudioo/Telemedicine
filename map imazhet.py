import pandas as pd
import os
import shutil

def lidh_imazhet_me_diagnoze():
    # 1. Ngarkojmë datasetin tonë dhe atë të NIH
    df_pacientet = pd.read_csv('dataset_diploma.csv')
    # Sigurohu që e ke shkarkuar këtë nga NIH
    df_nih = pd.read_csv('Data_Entry_2017.csv') 

    path_burimi = "images/" # Ku ndodhen imazhet e NIH
    path_destinacioni = "imazhet_e_pacienteve/"
    
    if not os.path.exists(path_destinacioni):
        os.makedirs(path_destinacioni)

    # Ndajmë imazhet e NIH në dy grupe: me probleme kardiake dhe normale
    nih_anomali = df_nih[df_nih['Finding Labels'].str.contains('Cardiomegaly|Effusion|Infiltration')]['Image Index'].tolist()
    nih_normal = df_nih[df_nih['Finding Labels'] == 'No Finding']['Image Index'].tolist()

    count = 0
    for _, row in df_pacientet.iterrows():
        p_id = row['Patient_ID']
        is_anomalous = row['Label'] # 1 ose 0 nga Random Forest ynë
        
        # Zgjedhim një imazh që i përshtatet diagnozës
        if is_anomalous == 1 and nih_anomali:
            img_to_copy = nih_anomali.pop(0)
        elif nih_normal:
            img_to_copy = nih_normal.pop(0)
        else:
            continue

        # Kopjojmë dhe riemërojmë imazhin
        shutil.copy(os.path.join(path_burimi, img_to_copy), 
                    os.path.join(path_destinacioni, f"{p_id}.png"))
        count += 1

    print(f"✅ U lidhën {count} imazhe me pacientët sipas diagnozës!")

lidh_imazhet_me_diagnoze()
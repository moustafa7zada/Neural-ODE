import pandas as pd 
import numpy as np
import time 

column_names = ["particle_id" ,"hit num1" ,"hit num2" , "hit num3","hit num4","hit num5","hit num6","hit num7","hit num8"
                ,"hit num9","hit num10","hit num11","hit num12","hit num13","hit num14","hit num15","hit num16"
                ,"hit num17","hit num18"]

  

def parser(left=1000 ,  right= 2000):
    for file in range(left , right) : 
            
        hits_file = pd.read_csv(f"event00000{file}-hits.csv")
        truth_file = pd.read_csv(f"event00000{file}-truth.csv")
        Training_data  = pd.DataFrame(np.nan , index=range(len(truth_file)) , columns=column_names)  
        appending_index = 0
        start  = time.time()
    
        for (_, hits_row), (_, true_row) in zip(hits_file.iterrows(), truth_file.iterrows()): 
            try :
                row_values  = Training_data.loc[Training_data["particle_id"] == true_row["particle_id"]].values[1]
                for index , cell_value in enumerate(row_values) :                 
                    if pd.isna(cell_value) : 
                        Training_data.loc[Training_data["particle_id"] == true_row["particle_id"] , f"hit num{index}"] = str(hits_row[["x", "y", "z"]].tolist())
                        break 
                    if index == 18 :
                        Training_data.loc[Training_data["particle_id"] == true_row["particle_id"] , f"hit num{index+1}"] = str(hits_row[["x", "y", "z"]].tolist())
            except :# if this if works , then its the first time we see the particl
                Training_data.loc[appending_index , ("particle_id" , "hit num1")] = (true_row["particle_id"] , str(hits_row[["x", "y", "z"]].tolist()) )
                appending_index += 1 
                
            
        
            
        print("Time for one hit" , time.time() - start)
        Training_data.to_csv(f"Training-data00000{file}.csv")
    
    
    print("Success! Good Luck insha Allah to CERN!")
    
    
    
    
    
    
    

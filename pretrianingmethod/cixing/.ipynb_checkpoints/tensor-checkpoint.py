import json

import numpy as np



def numb():
    dataList = []
    datas=[]
    poslist=["ADJ","ADP","PUNCT","ADV","AUX",
             "SYM","INTJ","CCONJ","X","NOUN",
             "DET","PROPN","NUM","VERB","PART",
             "PRON","SCONJ"]
    file_name="cixing.json"
    dataFilePath='tupos.json'
    with open(dataFilePath, 'r') as json_file:
        dataList=(json.load(json_file))
    
    
    a = np.zeros((18000,78))    
    for i in range(0,1800):
        a[i][0]=101
        
        
        for j in range(0,(len(dataList[i]))):
            
            
            # print(j)
            for q in range(0,17):
                
                print(q)
                if dataList[i][j] == poslist[q] and q!=2:
                    
                    a[i][j+1] = q+1
                    
        a[i][len(dataList[i])+1]=0
        
                    
    datas=a.tolist()    
    with open(file_name,'w') as file_object:
        json.dump(datas,file_object)
        
        

      
        
    
    

def main():
    numb()
        
if __name__ == "__main__":
    main()
    exit(0)
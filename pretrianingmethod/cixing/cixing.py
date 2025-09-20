import json





def cixing():
    dataList = []
    datas=[]
    file_name='tupos.json'
    dataFilePath='MCIDpos.json'
    with open(dataFilePath, 'r') as json_file:
        dataList.append(json.load(json_file))
        
    for i in range(0,9003):
        a=dataList[0][i]
        
        datas.append(a['tupos'])
      
        
    
    
    
    with open(file_name,'w') as file_object:
        json.dump(datas,file_object)
        
def main():
    cixing()
        
if __name__ == "__main__":
    main()
    exit(0)
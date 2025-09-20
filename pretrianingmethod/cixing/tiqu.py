

import json





def hebing():
    datas=[]


    dataFilePath='dataset.json'
    with open(dataFilePath, 'r') as json_file:
        dataList1=(json.load(json_file))
        

    b1=dataList1["stackoverflow"]
    t=0

    c1=b1["STACKOVERFLOW"]
    
   
    # 定义要忽略的符号列表
    symbols_to_ignore = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

    # 将文本中的符号替换为空格
    for i in range(0,len(c1)):
        for symbol in symbols_to_ignore:
            # print(c1[i][0])
            text1=c1[i][0]
        
            text = text1.replace(symbol, " ")
        datas.append(text)

            
            
            

    file_name='1.json'
    with open(file_name,'w') as file_object:
        json.dump(datas,file_object)
    # c2=b2['tword']
    # c3=b3['tword']
    # print((c1[10][0]))
    # print(len(c2[0]))
    
def main():
    hebing()
        
if __name__ == "__main__":
    main()
    exit(0)
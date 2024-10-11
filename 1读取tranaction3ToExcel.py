
import pandas as pd
import json
file = 'transactions 3.txt'
data = { }
list_s = []
with open(file,"r",encoding="utf-8") as f :
    for line in f.readlines() :
        line = line.strip("\n")
        list_s.append(line)
print(len(list_s))

# print(list_s[:2])
ls = []
for str in list_s[:5000] :
    di = json.loads(str)
    ls.append(di)

# 如果电脑内存太小，
# 就先读取20000条数据跑一下 ，pd1 = pd.DataFrame(ls[:20000)
pd1 = pd.DataFrame(ls)
pd1.to_excel('data1.xlsx')
print(f'{len(ls)}  data have insert excel  name is data.xlsx ---')










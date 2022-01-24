import  sys
import csv
import modelGet
sys.path.insert(1, 'src')
from src.model import *

state_dict=modelGet.main()
# open a file, where you stored the pickled data
file = open("checkpoints/"+"t1_s1_baseline_toy.pickle", 'rb')

# dump information to that file
data = pickle.load(file)

print("Items are:")
list = [["item id","item index","item embedding"]]
counter=0
for i in data.item_id_index.items():
    list.append([])
    print(i)
    list[counter+1].append(i)
    list[counter+1].append(state_dict[counter])
    counter +=1

print("Users are: ")
for i in data.user_id_index.items():
    print(i)

with open(f'pickleS1p1.tsv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    # write the data
    for i in range(len(list)):
      writer.writerow(list[i])
    print("File has been written")

#print(objects.item_id_index.items())
#print(objects.user_id_index.items())

file.close()
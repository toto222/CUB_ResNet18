import os
import random


data_list = list(range(1, 11789))


random.shuffle(data_list)


split_index = int(0.8 * len(data_list))


train_data = data_list[:split_index]
test_data = data_list[split_index:]

path = os.path.join('./dataset/CUB_200_2011',"data_split.txt")
with open(path, "w") as file:
    for data in train_data:
        file.write(f"{data} 1\n")  

    for data in test_data:
        file.write(f"{data} 0\n")  

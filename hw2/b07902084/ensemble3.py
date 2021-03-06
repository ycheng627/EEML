import numpy as np
from collections import Counter
import pandas as pd

files = [
    "prediction-0.7604823038709521.csv",
    "prediction-0.7612343791923117.csv",
    "prediction-0.7756832678149659.csv",
    "prediction-0.7750653489818567.csv",
    "prediction-0.7745734530686581.csv"
]

np_list = []
for i in files:
    np_arr = np.genfromtxt(i, skip_header=1, delimiter=",", dtype="int32")
    np_list.append(np_arr)


def get_mode(candidates):
    return Counter(candidates).most_common(1)[0]

    

# print(np_list[0].shape)
ans = np.copy(np_list[0])
for index in range(np_list[0].shape[0]):
    # print(np_list[0][index][0], end="\t")
    id = np_list[0][index][0]
    candidates = [arr[index][1] for arr in np_list]
    mode = get_mode(candidates)
    # print(mode)
    if mode[1] > 1:
        # print(mode[1])
        ans[index, 1] = mode[0]
    else:
        if ans[index-1, 1] in candidates: # when everyone is bad, assume continue from prev
            ans[index, 1] = ans[index-1, 1]
        else:
            ans[index, 1] = candidates[0]
    
    # print(ans[index])


a = pd.DataFrame(ans)
a.columns=["Id","Class"]
print(a.head(20))

a.to_csv("ensemble3.csv", index=False)


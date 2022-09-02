import pandas as pd 
import numpy as np 
import cvxpy as cp 



flip_df = pd.read_csv('denied_flipset.csv')

flip_df['ref_index'] = flip_df['Unnamed: 0']
flip_df = flip_df.drop(['Unnamed: 0'], axis=1)

#flip_df


#data processing
raw_df = pd.read_csv('german_raw.csv' , index_col = 0)
processed_df = pd.DataFrame(raw_df)

#processed_df['GoodCustomer'] = processed_df.index

category = pd.unique(processed_df['PurposeOfLoan']).tolist()

dic = {}

for i, name in enumerate(category):
    dic[name] = i   


processed_df['Gender'] = processed_df['Gender'].apply(lambda x: 1 if x== "Male" else -1)
processed_df['PurposeOfLoan'] = processed_df['PurposeOfLoan'].apply(lambda x: dic[x])
processed_df = processed_df.reset_index(drop=True)


#--------------------------------------------------
allocate_df = pd.DataFrame()
allocate_df['Gender'] = processed_df['Gender']


cost_dic = {key: 1000 for key in range(1000)}
for idx, val in zip(flip_df['ref_index'].tolist(), flip_df['total_cost']):
    #print(idx, val)
    cost_dic[idx] = val 
    

allocate_df['total_cost'] = pd.Series(cost_dic)
allocate_df['GoodCustomer'] = 1

# CVXPY Solution 

B = {}
for b in range(100):
    x = cp.Variable(len(allocate_df), boolean= True)
    #cons1 = allocate_df['total_cost'].to_numpy() @ x 
    #cons2 = allocate_df['Gender'].to_numpy() @ x  
    ones = np.ones(len(allocate_df))
    
    prob = cp.Problem(cp.Maximize(ones@x),
                     [allocate_df['total_cost'].to_numpy() @ x <= b, 
                     allocate_df['Gender'].to_numpy() @ x  == 0.0])
    prob.solve()
    
    # Print result.
    #print("\nThe optimal value is", prob.value)
    #print("A solution x is")
    #print(x.value)
    #print("A dual solution is")
    #print(prob.constraints[0].dual_value)

    if prob.value :
        B[b] = prob.value
        
print(B)

"""
{1: 32.0, 2: 52.0, 3: 70.0, 4: 88.0, 5: 102.0, 6: 114.0, 
 7: 126.0, 8: 138.0, 9: 148.0, 10: 160.0, 11: 172.0, 
 12: 182.0, 13: 190.0, 14: 198.0, 15: 206.0, 16: 214.0,
 17: 222.0, 18: 230.0, 19: 236.0, 20: 244.0, 21: 250.0, 
 22: 256.0, 23: 262.0, 24: 268.0, 25: 272.0, 26: 278.0,
 27: 284.0, 28: 290.0, 29: 296.0, 30: 300.0, 31: 306.0,
 32: 310.0, 33: 316.0, 34: 320.0, 35: 324.0, 36: 328.0, 
 37: 332.0, 38: 336.0, 39: 340.0, 40: 344.0, 41: 348.0, 
 42: 352.0, 43: 354.0, 44: 358.0, 45: 360.0, 46: 364.0, 
 47: 366.0, 48: 368.0, 49: 372.0, 50: 374.0, 51: 376.0, 
 52: 378.0, 53: 380.0, 54: 382.0, 55: 386.0, 56: 388.0, 
 57: 390.0, 58: 392.0, 59: 392.0, 60: 394.0, 61: 394.0, 
 62: 394.0, 63: 394.0, 64: 394.0, 65: 394.0, 66: 394.0, 
 67: 394.0, 68: 394.0, 69: 394.0, 70: 394.0, 71: 394.0, 
 72: 394.0, 73: 394.0, 74: 394.0, 75: 394.0, 76: 394.0, 
 77: 394.0, 78: 394.0, 79: 394.0, 80: 394.0, 81: 394.0, 
 82: 394.0, 83: 394.0, 84: 394.0, 85: 394.0, 86: 394.0,
 87: 394.0, 88: 394.0, 89: 394.0, 90: 394.0, 91: 394.0, 
 92: 394.0, 93: 394.0, 94: 394.0, 95: 394.0, 96: 394.0,
 97: 394.0, 98: 394.0, 99: 394.0}
"""



from edit import make_store_matrix
from predict import predict
import numpy as np

def exhaustive_search(n):
    step = round(1/n, 4)
    weights = np.arange(0 + step, 1 , step)
    weights[n-1] = 1
    optimum_costs = [0, 0, 0]
    max_auc = float("-inf")
    for icost in  weights:
        for dcost in weights:
            for rcost in weights:
                make_store_matrix(icost, dcost, rcost)

                auc, aupr = predict([f'--method={"wnngip"}',
                                    f'--dataset={"nr"}',
                                    f'--specify-arg={1}' 
                                            ])
                if auc > max_auc:
                    optimum_costs = [icost, dcost, rcost]
                    max_auc = auc
    print(f'weights : {optimum_costs} \n auc : {max_auc}')                
                
exhaustive_search(3)
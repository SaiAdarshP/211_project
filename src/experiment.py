from edit import make_store_matrix
from predict import predict

make_store_matrix(0.1, 0.1, 0.1)

auc, aupr = predict([f'--method={"wnngip"}',
                     f'--dataset={"nr"}',
                     f'--specify-arg={1}'
                             ])

print(auc)
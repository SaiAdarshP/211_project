import os


PATH = "../data/nr"
PATH2 = "../data/datasets"

def read_into_fileInfo():
    fileInfo = []
    for file in os.listdir(PATH):
        with open(os.path.join(PATH, file), 'r') as f:
            fileInfo.append((file.split('.')[0], f.read().rstrip()))
    return fileInfo


def weighted_edit_similarity(str1, str2, icost, dcost, rcost):
    m = len(str1)
    n = len(str2)
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i][j-1] + icost, dp[i-1][j] + dcost, dp[i-1][j-1] + rcost) 
 
    return 1 - dp[m][n]/max(m, n)

def make_store_matrix(icost, dcost, rcost):
    fileInfo = read_into_fileInfo()
    
    matrix = [[-1 for i in range(len(fileInfo)+1)] for i in range(len(fileInfo)+1)]
    matrix[0][0] = "data"

    for i, file1 in enumerate(fileInfo):
        for j, file2 in enumerate(fileInfo):
            matrix[i+1][j+1] = round(weighted_edit_similarity(file1[1], file2[1], icost, dcost, rcost), 5)
    for i in range(1, len(fileInfo)+1):
        matrix[i][0], matrix[0][i] = fileInfo[i-1][0], fileInfo[i-1][0]
    
    with open(os.path.join(PATH2, "nr_simmat_dc.txt"), 'w') as f:
        for line in matrix:
            for word in line:
                f.write(str(word))
                f.write('\t')
            f.write("\n")

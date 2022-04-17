from data.dataset import Dataset


d = Dataset("train", True, (0, 1), 5)

for X, y in d:
    print(X.ravel(), y.ravel())
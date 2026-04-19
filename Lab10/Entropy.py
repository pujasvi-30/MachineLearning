import pandas as pd
import math

def entropy(y):
    probs = y.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in probs)

def information_gain(data, column):
    total_entropy = entropy(data.iloc[:,-1])
    weighted_entropy = sum(
        (len(subset) / len(data)) * entropy(subset.iloc[:, -1])
        for _, subset in data.groupby(column)
    )
    return total_entropy - weighted_entropy

def main():
    data = pd.read_csv('playtennis.csv')
    target = data.columns[-1]
    print("Total Entropy:", entropy(data[target]))
    print("\nInformation Gain:\n")
    for col in data.columns[:-1]:
        print(f"{col}: {information_gain(data, col)}")


if __name__ == "__main__":
    main()
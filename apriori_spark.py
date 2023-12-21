from pyspark import SparkContext
from collections import defaultdict
import time

def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        for line in file:
            transaction = line.strip().split(',')
            dataset.append(set(transaction))
    return dataset

def generate_candidates(frequent_itemsets, k):
    candidates = []
    for itemset1 in frequent_itemsets:
        for itemset2 in frequent_itemsets:
            union_set = itemset1.union(itemset2)
            if len(union_set) == k:
                candidates.append(union_set)
    return candidates

def count_support(candidate, dataset):
    return sum(1 for transaction in dataset if candidate.issubset(transaction))

if __name__ == "__main__":
    dataset = load_dataset("data.csv")
    min_support = 3

    start_time = time.time()

    sc = SparkContext("local", "AprioriSpark")

    try:
        item_counts = defaultdict(int)
        for transaction in dataset:
            for item in transaction:
                item_counts[item] += 1

        frequent_itemsets = [set([item]) for item, count in item_counts.items() if count >= min_support]
        all_frequent_itemsets = frequent_itemsets

        k = 2
        while frequent_itemsets:
            candidate_itemsets = generate_candidates(frequent_itemsets, k)

            candidate_rdd = sc.parallelize(candidate_itemsets)

            item_counts = candidate_rdd.map(lambda candidate: (candidate, count_support(candidate, dataset)))
            frequent_itemsets = item_counts.filter(lambda item_count: item_count[1] >= min_support).map(lambda item_count: item_count[0]).collect()

            all_frequent_itemsets.extend(frequent_itemsets)
            k += 1

        print("Frequent Itemsets:")
        for itemset in all_frequent_itemsets:
            print(itemset)

    finally:
        sc.stop()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed Time:", elapsed_time, "seconds")

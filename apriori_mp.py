from collections import defaultdict
from mpi4py import MPI
import time

def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            transaction = line.strip().split(',')
            dataset.append(frozenset(transaction[1:]))
    return dataset

def generate_candidates(frequent_itemsets, k):
    return [itemset1.union(itemset2) for itemset1 in frequent_itemsets for itemset2 in frequent_itemsets if len(itemset1.union(itemset2)) == k]

def count_support(candidate, dataset):
    return sum(1 for transaction in dataset if candidate.issubset(transaction))

def mine_frequent_itemsets(dataset, min_support):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    item_counts = defaultdict(int)
    for transaction in dataset:
        for item in transaction:
            item_counts[item] += 1

    frequent_itemsets = [frozenset([item]) for item, count in item_counts.items() if count >= min_support]
    all_frequent_itemsets = frequent_itemsets

    k = 2
    start_time = time.time()

    while frequent_itemsets:
        if rank == 0:
            candidate_itemsets = generate_candidates(frequent_itemsets, k)
        else:
            candidate_itemsets = None

        candidate_itemsets = comm.bcast(candidate_itemsets, root=0)

        local_itemsets = [itemset for i, itemset in enumerate(candidate_itemsets) if i % size == rank]

        local_support = defaultdict(int)
        for itemset in local_itemsets:
            local_support[itemset] = count_support(itemset, dataset)

        global_support = comm.gather(local_support, root=0)

        if rank == 0:
            combined_support = defaultdict(int)
            for support in global_support:
                for itemset, count in support.items():
                    combined_support[itemset] += count

            frequent_itemsets = [itemset for itemset, count in combined_support.items() if count >= min_support]
            all_frequent_itemsets.extend(frequent_itemsets)
            k += 1

    if rank == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time

        print("Frequent Itemsets:")
        for itemset in all_frequent_itemsets:
            print(itemset)

        print("Elapsed Time:", elapsed_time, "seconds")

if __name__ == "__main__":
    dataset = load_dataset("data.csv")
    min_support = 3
    mine_frequent_itemsets(dataset, min_support)


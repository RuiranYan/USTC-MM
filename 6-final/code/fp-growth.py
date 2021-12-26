import time
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
# groceries = pd.read_csv("Groceries_dataset.csv")
groceries = pd.read_csv("trial.csv")

# Get all the transactions as a list of lists
all_transactions = [transaction[1]['itemDescription'].tolist()
                    for transaction in list(groceries.groupby(['Member_number', 'Date']))]
trans_encoder = TransactionEncoder()    # Instanciate the encoder
trans_encoder_matrix = trans_encoder.fit(all_transactions).transform(all_transactions)
trans_encoder_matrix = pd.DataFrame(trans_encoder_matrix, columns=trans_encoder.columns_)

# Find Frequent itemsets
frequent_itemsets = fpgrowth(trans_encoder_matrix, min_support=0.001, use_colnames=True)
print(frequent_itemsets)
# Generate Rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.02)
# sort
rules = rules.sort_values(['confidence'], ascending=False)
print(rules)

def perform_rule_calculation(transact_items_matrix, rule_type="fpgrowth", min_support=0.001):
    start_time = 0
    total_execution = 0
    if (not rule_type == "fpgrowth"):
        start_time = time.time()
        rule_items = apriori(transact_items_matrix,
                             min_support=min_support,
                             use_colnames=True, low_memory=True)
        total_execution = time.time() - start_time
        print("Computed Apriori!")
    else:
        start_time = time.time()
        rule_items = fpgrowth(transact_items_matrix,
                              min_support=min_support,
                              use_colnames=True)
        total_execution = time.time() - start_time
        print("Computed Fp Growth!")
    return total_execution

n_range = range(1, 10, 1)
list_time_ap = []
list_time_fp = []
for n in n_range:
    # (time_ap, time_fp) = (0, 0)
    min_sup = float(n / 100)
    time_ap = perform_rule_calculation(
        trans_encoder_matrix, rule_type="fpgrowth", min_support=min_sup)
    time_fp = perform_rule_calculation(
        trans_encoder_matrix, rule_type="aprior", min_support=min_sup)
    list_time_ap.append(time_ap)
    list_time_fp.append(time_fp)

plt.plot(n_range, list_time_ap, label='Apriori', color='green')
plt.plot(n_range, list_time_fp, label='Fp_growth', color='red')
plt.xlabel("Support (%)")
plt.ylabel("Run Time (seconds)")
plt.title("time: Apriori vs Fp_growth")
plt.legend(loc="best")
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px

#data = pd.read_csv("Groceries_dataset.csv")
data = pd.read_csv("trial.csv")
transactions = [a[1]['itemDescription'].tolist()
                for a in list(data.groupby(['Member_number', 'Date']))]
print(transactions[:10])
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)   # encode to bool values
transactions = pd.DataFrame(te_ary, columns=te.columns_)  
print(transactions)  
freq_items = apriori(transactions, min_support=0.001, use_colnames=True, verbose=1)
freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))
#freq_items = freq_items.sort_values(['support'], ascending=False)
print(freq_items.head(10))
print(freq_items.tail(10))      
rules = association_rules(freq_items, metric="confidence", min_threshold=0.001)
print(rules.head())
fig = px.scatter(rules['support'], rules['confidence'])
fig.update_layout(
    xaxis_title="support",
    yaxis_title="confidence",
    font_family="Courier New",
    font_color="blue",
    title_font_family="Times New Roman",
    title_font_color="red",
    title=('Support vs Confidence')
)
fig.show()

fig = px.scatter(rules['support'], rules['lift'])
fig.update_layout(
    xaxis_title="support",
    yaxis_title="lift",
    font_family="Courier New",
    font_color="blue",
    title_font_family="Times New Roman",
    title_font_color="red",
    title=('Support vs Lift')
)
fig.show()

fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], '.', rules['lift'], fit_fn(rules['lift']))
plt.xlabel('lift')
plt.ylabel('Confidence')
plt.title('lift vs Confidence')
plt.show()
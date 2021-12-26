import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder
import seaborn as sns
from wordcloud import WordCloud

# figure: Total Number of Items Sold by Date
data = pd.read_csv("Groceries_dataset.csv")
# data = pd.read_csv("trial.csv")
data.groupby(['Date'])['itemDescription'].agg(['count'])\
    .plot(figsize=(12, 5), grid=True, title="Total Number of Items Sold by Date")\
    .set(xlabel="Date", ylabel="Total Number of Items Sold")
plt.show()

d = data.set_index(['Date'])
d.index = pd.to_datetime(d.index)
# figure: Total Number of Items Sold by Month
d.resample("M")['itemDescription'].count().\
    plot(figsize=(12, 5), grid=True, title="Total Number by Items Sold by Month").\
    set(xlabel="Date", ylabel="Total Number of Items Sold")
plt.show()

# watch out the statistics of the dataset
total_items = len(d)
total_days = len(np.unique(d.index.date))
total_months = len(np.unique(d.index.month))
average_items = total_items / total_days
unique_items = d.itemDescription.unique().size
print("There are {} unique items sold ".format(unique_items))
print("Total {} items sold in {} days throughout {} months".format(total_items, total_days, total_months))
print("With an average of {} items sold daily".format(int(average_items)))
print(data['itemDescription'].value_counts())

# visualize the dataset
def bar_plot(df, col):
    fig = px.bar(df,
                 x=df[col].value_counts().keys(),
                 y=df[col].value_counts().values,
                 color=df[col].value_counts().keys())
    fig.update_layout(
        xaxis_title=col,
        yaxis_title="Count",
        legend_title=col,
        font_family="Courier New",
        font_color="blue",
        title_font_family="Times New Roman",
        title_font_color="red",
        legend_title_font_color="green")
    fig.show()
# bar_plot(data, 'itemDescription')

# group by member/name
df = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(sum)
transactions = [a[1]['itemDescription'].tolist()
                for a in list(data.groupby(['Member_number', 'Date']))]
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)   # encode to bool values

transactions = pd.DataFrame(te_ary, columns=te.columns_)
pf = transactions.describe()
f = pf.iloc[0] - pf.iloc[3]
a = f.tolist()
b = list(f.index)
item = pd.DataFrame([[a[r], b[r]]for r in range(len(a))], columns=['Count','Item'])
item = item.sort_values(['Count'], ascending=False).head(50)
print(item)     # top 50

plt.rcParams['figure.figsize'] = (15, 15)
wordcloud = WordCloud(background_color='white', width=1200, height=1200, max_words=121)\
    .generate(str(item['Item']))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Items', fontsize=20)
# plt.show()
# fig = px.treemap(item, path=['Item'], values='Count')
# fig.show()
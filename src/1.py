from itertools import count
import pandas as pd
import matplotlib as plt


print(pd.__version__)

df = pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
df = pd.DataFrame(
    {'Bob': ['I liked it.', 'It was awful'],
        'Sue': ['Pretty good.', 'Bland.']},
    index=['Product A', 'Product B'])
df

pd.Series([1, 2, 3, 4, 5])

pd.Series([30, 35, 40], index=['2015 Sales',
          '2016 Sales', '2017 Sales'], name='Product A')

# wine_reviews = pd.read_csv("data/winemag-data-130k-v2.csv")
reviews = pd.read_csv("../data/winemag-data-130k-v2.csv", index_col=0)
reviews
reviews.shape
reviews.head()

pd.set_option('max_rows', 5)

reviews.country
reviews['country']

reviews.iloc[0]
type(reviews.iloc[0])

reviews.iloc[:, 0]
reviews.iloc[:3, 0]
reviews.iloc[1:3, 0]
reviews.iloc[[0, 1, 2], 1]
reviews.loc[0, 'country']
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]

reviews.set_index("title")
reviews.set_index("description")
reviews.set_index("country")
reviews.loc[:5, ['designation', 'points']]

reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]
reviews.loc[reviews.country.isin(['Italy', 'France'])]
reviews.loc[reviews.price.notnull()]
reviews['critic'] = 'everyone'
reviews['index_backwards'] = range(len(reviews), 0, -1)
reviews['index_backwards']

top_oceania_wines = reviews.loc[reviews.isin(
    ['Australia', 'New Zealand']) | (reviews.points >= 95)]

reviews.points.describe()

reviews_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - reviews_points_mean)


def remean_points(row):
    row.points = row.points - reviews_points_mean
    return row


reviews.apply(remean_points, axis='columns')

reviews.head(1)

reviews.country + " - " + reviews.region_1


def to_star(row):
    if row.points >= 95 or row.country == "Canada":
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1


# replace value_counts() with groups
reviews.groupby('points').points.count()
reviews.groupby('points').price.min()

# pick best wine by country and province with double grouping
reviews.groupby(['country', 'province']).apply(
    lambda df: df.loc[df.points.idxmax()])

# use agg to run different functions simultaneously
reviews.groupby(['country']).price.agg([len, min, max])

# multi-index
countries_reviewed = reviews.groupby(
    ['country', 'province']).description.agg([len])
countries_reviewed

mi = countries_reviewed.index
type(mi)

# sorting values
countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len', ascending=False)
countries_reviewed.sort_index()
countries_reviewed.sort_values(by=['country', 'len'], ascending=False)

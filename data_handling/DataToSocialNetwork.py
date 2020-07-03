#reads file in the form of uid - iid - rating and transforms it to a social network
#two files written: 1/ with counts of co-ratings 2/ with normalized relative counts of co-ratings
#
#
#specify input file in filepath_read
#specify output file for social network with counts in filepath_write
#specify output file for social network with counts in filepath_write_normalize
#
#author: Peter Dolog, Aalborg University

import pandas as pd
from itertools import combinations


filepath_read = "ciao_DVD_train_data_25.txt"
filepath_write = "socialnetwork.csv"
filepath_write_normalize = "socialnetwork_normalized.csv"


data = pd.read_csv(filepath_read)

#print(data.head())

#absolute counts
#counting how many have co-rated
rawcounts = pd.value_counts(
    [(x, y) for _, d in data.groupby('iid') for x, y in combinations(d.uid, 2)]
)

#getting combinations from index to friends
socnetwork = rawcounts.reset_index()

#putting it into dataframe again
socnetwork = pd.DataFrame(socnetwork)

#print(socnetwork)

#formatting for appending to a file
#changing numeric values of friends pairs to string
socnetwork['friends'] = socnetwork['index'].astype(str)

#print(socnetwork)

#removing brackets from friends column
socnetwork['friends'] = socnetwork['friends'].str.replace(r'(', '')
socnetwork['friends'] = socnetwork['friends'].str.replace(r')', '')

#print(socnetwork)

#splitting the pairs into separate columns
socnetwork[['uid1', 'uid2']] = socnetwork.friends.str.split(", ", expand=True)

#removing temporal columns
socnetwork = socnetwork.drop(columns=['index', 'friends'])


#renaming column with counts to co-ratings
socnetwork = socnetwork.rename(columns={0:"co-ratings"})

print(socnetwork.head())

#writing data frame to file
socnetwork.to_csv(filepath_write, index = False)

#relative counts
#counting how many have co-rated
rawcounts_n = pd.value_counts(
    [(x, y) for _, d in data.groupby('iid') for x, y in combinations(d.uid, 2)], normalize=True
)

#getting combinations from index to friends
socnetwork_n = rawcounts_n.reset_index()

#putting it into dataframe again
socnetwork_n = pd.DataFrame(socnetwork_n)

#print(socnetwork)

#formatting for appending to a file
#changing numeric values of friends pairs to string
socnetwork_n['friends'] = socnetwork_n['index'].astype(str)

#print(socnetwork)

#removing brackets from friends column
socnetwork_n['friends'] = socnetwork_n['friends'].str.replace(r'(', '')
socnetwork_n['friends'] = socnetwork_n['friends'].str.replace(r')', '')

#print(socnetwork)

#splitting the pairs into separate columns
socnetwork_n[['uid1', 'uid2']] = socnetwork_n.friends.str.split(", ", expand=True)

#removing temporal columns
socnetwork_n = socnetwork_n.drop(columns=['index', 'friends'])


#renaming column with counts to co-ratings
socnetwork_n = socnetwork_n.rename(columns={0:"co-ratings"})

print(socnetwork_n.head())


#writing data frame to file with normalized relative numbers
socnetwork_n.to_csv(filepath_write_normalize, index = False)
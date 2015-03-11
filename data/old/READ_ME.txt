Data description:

The dataset rb_adultcontemp_withduplicates.csv contains all lyrical information for genres R and B and Adult contemporary. This includes duplicates when songs occur over multiple weeks on the Billboard charts. This dataset includes all columns as well as missing lyrics fields. It is not to be used for model building, but rather only for reference if needed.

****************************************************************************************

The dataset rb_adultcontemp_train.csv contains only unique row entries from the previous dataset, with a random subset representing 5% of the data removed for final testing purposes. Only includes columns ‘genre’, ‘artist’, ‘song’, ‘lyrics_body’, ‘year’. Entries have been removed which contain missing lyrics or are marked as instrumental.  

The dataset rb_adultcontemp_holdout.csv is the five percent random holdout sample from the above dataset. Same columns and same subsetting restrictions.
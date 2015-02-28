# merges lyrics form last semester project with dataset from this semester. 

# read in total dataset:
total <- read.csv('~/Desktop/MSAN/Module3/ML2/project/total.csv')

# subset to genre RandB and Adult Contemp:
df1 <- total[total$genre=='R_and_B' | total$genre=='adult_contemp',]

# remove all instrumentals:
df2 <- df1[df1$lyrics_body != "INSTRUMENTAL",]

# subset to include only these 5 columns:
df3 <- subset(df2, select = c('genre', 'artist', 'song', 'lyrics_body', 'year'))

# remove all cases where lyrics are missing:
df4 <- df3[df3$lyrics_body != '',]

# create training and test sets:
n <- nrow(df4)
percent_train <- 0.95
samp_size <- floor(percent_train * n)
train_idx <- sample(seq_len(n),size=samp_size)
df_train <- df4[train_idx,]
df_test <- df4[-train_idx,]

# plot distributions of test and training genres:
plot(df_train$genre)
plot(df_test$genre)

# write datasets to csv files:
write.csv(df_train, 'rb_adultcontemp_train.csv')
write.csv(df_test, 'rb_adultcontemp_holdout.csv')


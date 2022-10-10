---
title: "Assigment - Naive Bayes DIY"
author:
  - Duyen Nguyen - Author
  - Thanh Dung Nguyen - Reviewer
date: "`r format(Sys.time(), '%d %B, %Y')`"
output:
  html_notebook:
  toc: true
toc_depth: 2
---
  
  ```{r}
library(tidyverse)
library(tm)
library(caret)
library(wordcloud)
library(e1071)
```
---
  
  Choose a suitable dataset from [this](https://github.com/HAN-M3DM-Data-Mining/assignments/tree/master/datasets) folder and train your own Naive Bayes model. Follow all the steps from the CRISP-DM model.


## Business Understanding
using Naive Bayes model to filter and handle fakenews

## Data Understanding

```{r}
rawDF <- NB_fakenews
str(rawDF)

```
## Data preparation
# remove first id variable
```{r}
cleanDF <- rawDF[-1]
head(cleanDF)
table(cleanDF$label)
summary(cleanDF)
```

# change to factor type
```{r}
cleanDF$label <- factor(cleanDF$label, levels = c("1", "0"), labels = c("Fakenews", "Non")) %>% relevel("Non")
head(cleanDF, 9)
```


# visually inspect the data by creating wordclouds
```{r}
Fakenews <- cleanDF %>% filter(label == "Fakenew")
Non <- cleanDF %>% filter(label == "Non")

wordcloud(Fakenews$text, max.words = 20, scale = c(4, 0.8), colors= c("indianred1","indianred2","indianred3","indianred"))
wordcloud(Non$text, max.words = 20, scale = c(4, 0.8), colors= c("lightsteelblue1","lightsteelblue2","lightsteelblue3","lightsteelblue"))
```

```{r}
rawCorpus <- Corpus(VectorSource(cleanDF$text))
inspect(rawCorpus[1:2])
```

# remove numbers, punctuation, unuseful words, and change to lower case
```{r}
cleanCorpus <- rawCorpus %>% 
  tm_map(tolower) %>% 
  tm_map(removeNumbers) %>% 
  tm_map(removeWords, stopwords()) %>% 
  tm_map(removePunctuation) %>% 
  tm_map(stripWhitespace)
```
# inspect the corpus
```{r}
tibble(Raw = rawCorpus$content[1:3], Clean = cleanCorpus$content[1:3])
```

# transform to matrix
```{r}
cleanDTM <- cleanCorpus %>% DocumentTermMatrix
inspect(cleanDTM)
```
# Create split indices
```{r}
set.seed(1234)
trainIndex <- createDataPartition(cleanDF$label, p = .75, 
                                  list = FALSE, 
                                  times = 2)
head(trainIndex)
```

# Apply split indices to DF
```{r}
trainDF <- cleanDF[trainIndex, ]
testDF <- cleanDF[-trainIndex, ]
```


# Apply split indices to Corpus
```{r}

trainCorpus <- cleanCorpus[trainIndex]
testCorpus <- cleanCorpus[-trainIndex]
```


# Apply split indices to DTM
```{r}

trainDTM <- cleanDTM[trainIndex, ]
testDTM <- cleanDTM[-trainIndex, 2]
```

# eliminate words with low frequencies
```{r}
freqWords <- trainDTM %>% findFreqTerms(5)
trainDTM <-  DocumentTermMatrix(trainCorpus, list(dictionary = freqWords))
testDTM <-  DocumentTermMatrix(testCorpus, list(dictionary = freqWords))
```

# transform the counts into a factor -> see whether the word appears in the document or not
```{r}
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0) %>% factor(levels = c(0,1), labels = c("No", "Yes"))
}
```

```{r}
nColsDTM <- dim(trainDTM)[2]
trainDTM <- apply(trainDTM, MARGIN = 2, convert_counts)
testDTM <- apply(testDTM, MARGIN = 2, convert_counts)
```

```{r}
head(trainDTM[,1:10])
```


## Modeling
```{r}
nbayesModel <-  naiveBayes(trainDTM, trainDF$type, laplace = 2)
```

```{r}
predVec <- predict(nbayesModel, testDTM)
confusionMatrix(predVec, testDF$type, positive = "Fakenews", dnn = c("Prediction", "True"))
```


## Evaluation and Deployment
text and code here

reviewer adds suggestions for improving the model
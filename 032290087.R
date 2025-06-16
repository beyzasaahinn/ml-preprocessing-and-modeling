### GÖREV 1 ###



library(readr)
data <- read_csv("ameshousing.csv")

head(data)

colSums(is.na(data))

sum(is.na(data))

install.packages("visdat")

library(visdat)
vis_miss(data)


# Sayısal sütunları seçmek
num_cols <- sapply(data, is.numeric)

# Sayısal sütunlarda eksik değerleri ortalama ile doldurmak
data_num <- data[, num_cols]
data_num[] <- lapply(data_num, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))

# Yeni veri setini birleştirmek
data_imputed_simple <- cbind(data_num, data[, !num_cols])



# Kategorik sütunları seçmek
cat_cols <- sapply(data, is.factor)

# Kategorik sütunlarda eksik değerleri mod ile doldurmak
data_cat <- data[, cat_cols]
data_cat[] <- lapply(data_cat, function(x) ifelse(is.na(x), names(sort(table(x), decreasing = TRUE))[1], x))

# Yeni veri setini birleştirmek
data_imputed_simple <- cbind(data_num, data_cat)




install.packages("VIM")
library(VIM)

# KNN imputasyonu VIM kütüphanesi ile yapma
data_imputed_knn_vim <- kNN(data, k = 5)



library(mice)

# Tree-based (cart) yöntemiyle eksik verileri doldurmak
data_imputed_tree <- mice(data, method = "cart", m = 1)  # m = 1: Tek bir tahmin kümesi
data_imputed_tree <- complete(data_imputed_tree)


sum(is.na(data_imputed_simple))



# Basit Doldurma Sonrası
par(mfrow = c(1, 2))
hist(data_imputed_simple$SalePrice, main = "Basit Doldurma - Histogram", col = "lightblue")
boxplot(data_imputed_simple$SalePrice, main = "Basit Doldurma - Boxplot", col = "lightgreen")

# KNN Imputation Sonrası
hist(data_imputed_knn$SalePrice, main = "KNN - Histogram", col = "lightblue")
boxplot(data_imputed_knn$SalePrice, main = "KNN - Boxplot", col = "lightgreen")

# Tree-based Imputation Sonrası
hist(data_imputed_tree$SalePrice, main = "Tree-based - Histogram", col = "lightblue")
boxplot(data_imputed_tree$SalePrice, main = "Tree-based - Boxplot", col = "lightgreen")




sum(is.na(data_imputed_simple))
sum(is.na(data_imputed_knn))
sum(is.na(data_imputed_tree))

#####################################################
#####################################################
#####################################################

### GÖREV 2 ###


library(tidyverse) 
library(caret)      
library(ggplot2)   

#install.packages('caret')


wine <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";")

head(wine)

str(wine)

sum(is.na(wine))

#kalite değişkenini çıkardık çünkü sadece özellikleri standardize etmek istiyoruz.
preprocess_params <- preProcess(wine[, -12], method = c("center", "scale"))

#standardizasyon
wine_scaled <- predict(preprocess_params, wine[, -12])

head(wine_scaled)

#varyans özellikleri belirleme
nzv <- nearZeroVar(wine_scaled, saveMetrics = TRUE)

#varyanslı olanları görüntüleyelim
nzv[nzv$nzv, ]

#near-zero varyanslı sütunlar varsa, bunları çıkartalım:
wine_filtered <- wine_scaled[, !nzv$nzv]

# PCA 
pca_model <- prcomp(wine_filtered, center = TRUE, scale. = TRUE)

# varyansın %95'ini koruyacak bileşen sayısını belirleme
var_explained <- cumsum(pca_model$sdev^2 / sum(pca_model$sdev^2))

#ilk kaç bileşen %95 
num_components <- which(var_explained >= 0.95)[1]

num_components

# PCA bileşenleri
pca_data <- as.data.frame(pca_model$x[, 1:2])

# görselleştirme
ggplot(pca_data, aes(x = PC1, y = PC2)) +
  geom_point(alpha = 0.5, color = "pink") +
  ggtitle("PCA: İlk İki Bileşen") +
  theme_minimal()


######################################################
######################################################
######################################################

### GÖREV 3 ###

library(caret)
library(e1071)        # SVM için
library(randomForest) # RF için
library(xgboost)

df <- read.csv("train.csv", stringsAsFactors = FALSE)

#NA Doldurma ; caret'in preProcess fonksiyonunu kullanarak sayısal sütunlarda medyan doldurma yapılabilir.

target <- "SalePrice"
predictorNames <- setdiff(names(df), c(target))

#kategorik sütunları dummy'ya çevireceğiz.
dmy <- dummyVars(" ~ .", data = df[, predictorNames])
df_processed <- data.frame(predict(dmy, newdata = df[, predictorNames]))

#preProcess ile NA doldurma uygulayalım
preProcValues <- preProcess(df_processed, method = c("medianImpute"))
df_imputed <- predict(preProcValues, df_processed)

# hedef değişkeni de df'den ekleyelim
df_imputed[[target]] <- df[[target]]

#hedef değişken dönüşümlerii
#Log dönüşüm
df_imputed$SalePrice_log <- log(df_imputed$SalePrice)

#Box-Cox dönüşümü
boxcoxObj <- BoxCoxTrans(df_imputed$SalePrice)
df_imputed$SalePrice_boxcox <- predict(boxcoxObj, df_imputed$SalePrice)

set.seed(123)
trainIndex <- createDataPartition(df_imputed[[target]], p = 0.8, list = FALSE)
trainData <- df_imputed[trainIndex, ]
testData  <- df_imputed[-trainIndex, ]

# modelde kullanılacak predictor değişkenler
# hedef ve dönüşüm sütunlarını çıkarıyoruz.
predictorCols <- setdiff(names(trainData), c("SalePrice", "SalePrice_log", "SalePrice_boxcox"))

#Model Eğitim ve Değerlendirme

##KNN 
set.seed(123)
model_knn <- train(trainData[, predictorCols],
                   trainData$SalePrice,
                   method = "knn",
                   trControl = trainControl(method = "cv", number = 3))
pred_knn <- predict(model_knn, testData[, predictorCols])
rmse_knn <- sqrt(mean((pred_knn - testData$SalePrice)^2))
cat("KNN (Ham Hedef) RMSE:", rmse_knn, "\n")

##Random Forest 
set.seed(123)
model_rf <- train(trainData[, predictorCols],
                  trainData$SalePrice_log,
                  method = "rf",
                  trControl = trainControl(method = "cv", number = 3))
pred_rf_log <- predict(model_rf, testData[, predictorCols])
pred_rf <- exp(pred_rf_log)  # Geri dönüşüm: log -> exp
rmse_rf <- sqrt(mean((pred_rf - testData$SalePrice)^2))
cat("Random Forest RMSE:", rmse_rf, "\n")

set.seed(123)
model_xgb <- train(trainData[, predictorCols],
                   trainData$SalePrice_boxcox,
                   method = "xgbTree",
                   trControl = trainControl(method = "cv", number = 3))
# test verisi için Box-Cox ölçekli tahmin
pred_xgb_boxcox <- predict(model_xgb, testData[, predictorCols])
# tahminleri orijinal ölçeğe geri dönüştürmek için Box-Cox ters dönüşümü
pred_xgb <- predict(boxcoxObj, pred_xgb_boxcox, inverse = TRUE)
rmse_xgb <- sqrt(mean((pred_xgb - testData$SalePrice)^2))
cat("XGBoost (Box-Cox Dönüştürülmüş Hedef) RMSE:", rmse_xgb, "\n")

# sonuçların Karşılaştırılması
rmse_results <- data.frame(
  Model = c("KNN (Ham Hedef)", "RF (Log Dönüştürülmüş)", "XGBoost (Box-Cox Dönüştürülmüş)"),
  RMSE = c(rmse_knn, rmse_rf, rmse_xgb)
)
print(rmse_results)
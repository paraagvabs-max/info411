library(dplyr)
library(rpart)
library(rpart.plot)
library(caret)
library(nnet)
library(pROC)

raw_data <- read.csv(file.choose())

df_clean <- raw_data %>%
  filter(!is.na(CustomerID)) %>%
  mutate(
    InvoiceDate = as.Date(InvoiceDate), 
    LineTotal = Quantity * UnitPrice
  )

max_date <- max(df_clean$InvoiceDate, na.rm = TRUE)

customer_data <- df_clean %>%
  group_by(CustomerID) %>%
  summarise(
    DaysSinceLastPurchase = as.numeric(max_date - max(InvoiceDate, na.rm = TRUE)),
    PurchaseFrequency = n_distinct(InvoiceNo),
    TotalSpent = sum(LineTotal),
    AvgItemPrice = mean(UnitPrice, na.rm = TRUE),
    TotalItemsBought = sum(Quantity, na.rm = TRUE),
    UniqueProducts = n_distinct(StockCode)
  ) %>%
  ungroup()

customer_data <- customer_data %>%
  mutate(
    IsChurned = ifelse(
      DaysSinceLastPurchase > 90 | (PurchaseFrequency <= 3 & TotalSpent <= 100),
      "Yes", 
      "No"
    )
  )

customer_data$IsChurned <- as.factor(customer_data$IsChurned)

model_data <- customer_data %>% 
  dplyr::select(AvgItemPrice, TotalItemsBought, UniqueProducts, IsChurned)

model_data <- na.omit(model_data)
model_data[, 1:3] <- scale(model_data[, 1:3])

set.seed(123)
ind <- sample(1:nrow(model_data), size = round(0.7 * nrow(model_data)))
train_data <- model_data[ind, ]
test_data <- model_data[-ind, ]

dt_model <- rpart(IsChurned ~ ., data = train_data, method = "class", control = rpart.control(cp = 0.001))

plotcp(dt_model)
best_cp <- dt_model$cptable[which.min(dt_model$cptable[,"xerror"]), "CP"]
dt_pruned <- prune(dt_model, cp = best_cp)

rpart.plot(dt_pruned, main = "Pruned Decision Tree: Customer Churn")

set.seed(123)
nn_model <- nnet(IsChurned ~ ., data = train_data, size = 5, decay = 0.1, maxit = 500)

dt_preds <- predict(dt_pruned, test_data, type = "class")
nn_preds <- factor(nn_preds, levels = levels(test_data$IsChurned))

print("--- Decision Tree Evaluation ---")
print(confusionMatrix(dt_preds, test_data$IsChurned, positive = "Yes"))

print("--- Neural Network EvSaluation ---")
print(confusionMatrix(nn_preds, test_data$IsChurned, positive = "Yes"))

dt_probs <- predict(dt_pruned, test_data, type = "prob")[, "Yes"]
nn_probs <- predict(nn_model, test_data, type = "raw")[, 1]

roc_dt <- roc(test_data$IsChurned, dt_probs, levels = c("No", "Yes"))
roc_nn <- roc(test_data$IsChurned, nn_probs, levels = c("No", "Yes"))

plot(roc_dt, col = "blue", main = "ROC Curve Comparison: DT vs NN", lwd = 2)
lines(roc_nn, col = "red", lwd = 2)
legend("bottomright", 
       legend = c(paste("Decision Tree (AUC =", round(auc(roc_dt), 3), ")"),
                  paste("Neural Network (AUC =", round(auc(roc_nn), 3), ")")),
       col = c("blue", "red"), lwd = 2)

set.seed(123)
nn_model_2 <- nnet(IsChurned ~ ., data = train_data, size = 15, decay = 0.001, maxit = 1000)

nn_preds_2 <- predict(nn_model_2, test_data, type = "class")

nn_preds_2 <- factor(nn_preds_2, levels = levels(test_data$IsChurned))

print("--- Neural Network 2 (Complex) Evaluation ---")
print(confusionMatrix(nn_preds_2, test_data$IsChurned, positive = "Yes"))
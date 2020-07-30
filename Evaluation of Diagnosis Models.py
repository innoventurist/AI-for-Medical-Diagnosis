
# coding: utf-8

# Evaluation of Diagnostic Models
# Goal:
# 1. Accuracy
# 1. Prevalence
# 1. Specificity & Sensitivity
# 1. PPV and NPV
# 1. ROC curve and AUCROC (c-statistic)
# 1. Confidence Intervals


# Import Packages
import numpy as np # popular library for scientific computing
import matplotlib.pyplot as plt # plotting library compatible with numpy
import pandas as pd  # used to manipulate data

import util



# Take a quick look at the dataset. The data is stored in two CSV files called `train_preds.csv` and `valid_preds.csv`.
train_results = pd.read_csv("train_preds.csv")
valid_results = pd.read_csv("valid_preds.csv")

# the labels in our dataset
class_labels = ['Cardiomegaly',
 'Emphysema',
 'Effusion',
 'Hernia',
 'Infiltration',
 'Mass',
 'Nodule',
 'Atelectasis',
 'Pneumothorax',
 'Pleural_Thickening',
 'Pneumonia',
 'Fibrosis',
 'Edema',
 'Consolidation']

# the labels for prediction values in our dataset
pred_labels = [l + "_pred" for l in class_labels]


# Extract the labels (y) and the predictions (pred).
y = valid_results[class_labels].values
pred = valid_results[pred_labels].values


# Run the next cell to view them side by side and look at the dataset.
valid_results[np.concatenate([class_labels, pred_labels])].head()


# To further understand our dataset details, here's a histogram of the number of samples for each label in the validation dataset:
plt.xticks(rotation=90)
plt.bar(x = class_labels, height= y.sum(axis=0));

#
# As the name suggests
# - true positive (TP): The model classifies the example as positive, and the actual label also positive.
# - false positive (FP): The model classifies the example as positive, **but** the actual label is negative.
# - true negative (TN): The model classifies the example as negative, and the actual label is also negative.
# - false negative (FN): The model classifies the example as negative, **but** the label is actually positive.


# Compute from the model predictions for true positives, true negatives, false positives, and false negatives
def true_positives(y, pred, th=0.5):
    """
    Count true positives.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        TP (int): true positives
    """
    TP = 0
    
    # get thresholded predictions
    thresholded_preds = pred >= th

    # compute TP
    TP = np.sum((y == 1) & (thresholded_preds == 1))
    
    return TP

def true_negatives(y, pred, th=0.5):
    """
    Count true negatives.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        TN (int): true negatives
    """
    TN = 0
    
    # get thresholded predictions
    thresholded_preds = pred >= th
    
    # compute TN
    TN = np.sum((y == 0) & (thresholded_preds == 0))
    
    return TN

def false_positives(y, pred, th=0.5):
    """
    Count false positives.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        FP (int): false positives
    """
    FP = 0
    
    # get thresholded predictions
    thresholded_preds = pred >= th

    # compute FP
    FP = np.sum((y == 0) & (thresholded_preds == 1))
    
    return FP

def false_negatives(y, pred, th=0.5):
    """
    Count false positives.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        FN (int): false negatives
    """
    FN = 0
    
    # get thresholded predictions
    thresholded_preds = pred >= th
    
    # compute FN
    FN = np.sum((y == 1) & (thresholded_preds == 0))
    
    
    return FN


# Note: must explicity import 'display' in order for the autograder to compile the code
from IPython.display import display
# Test
df = pd.DataFrame({'y_test': [1,1,0,0,0,0,0,0,0,1,1,1,1,1],
                   'preds_test': [0.8,0.7,0.4,0.3,0.2,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0],
                   'category': ['TP','TP','TN','TN','TN','FP','FP','FP','FP','FN','FN','FN','FN','FN']
                  })

display(df)
#y_test = np.array([1, 0, 0, 1, 1])
y_test = df['y_test']

#preds_test = np.array([0.8, 0.8, 0.4, 0.6, 0.3])
preds_test = df['preds_test']

threshold = 0.5
print(f"threshold: {threshold}\n")

print(f"""Our functions calculated: 
TP: {true_positives(y_test, preds_test, threshold)}
TN: {true_negatives(y_test, preds_test, threshold)}
FP: {false_positives(y_test, preds_test, threshold)}
FN: {false_negatives(y_test, preds_test, threshold)}
""")

print("Expected results")
print(f"There are {sum(df['category'] == 'TP')} TP")
print(f"There are {sum(df['category'] == 'TN')} TN")
print(f"There are {sum(df['category'] == 'FP')} FP")
print(f"There are {sum(df['category'] == 'FN')} FN")


# Run the next cell to see a summary of evaluative metrics for the model predictions for each class. 
util.get_performance_metrics(y, pred, class_labels) 


# Use a threshold of .5 to as a cutoff for predictions and calculare the model's accuracy
def get_accuracy(y, pred, th=0.5):
    """
    Compute accuracy of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        accuracy (float): accuracy of predictions at threshold
    """
    accuracy = 0.0
    
    # get TP, FP, TN, FN using our previously defined functions
    TP = true_positives(y, pred, th)
    FP = false_positives(y, pred, th)
    TN = true_negatives(y, pred, th)
    FN = false_negatives(y, pred, th)

    # Compute accuracy using TP, FP, TN, FN
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    
    return accuracy


# Test
print("Test case:")

y_test = np.array([1, 0, 0, 1, 1])
print('test labels: {y_test}')

preds_test = np.array([0.8, 0.8, 0.4, 0.6, 0.3])
print(f'test predictions: {preds_test}')

threshold = 0.5
print(f"threshold: {threshold}")

print(f"computed accuracy: {get_accuracy(y_test, preds_test, threshold)}")

# Run the next cell to see the accuracy of the model output for each class.
# And see the number of true positives, true negatives, false positives, and false negatives.
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy)



# 
# Calculate the accuracy for the model.
get_accuracy(valid_results["Emphysema"].values, np.zeros(len(valid_results)))


# In a medical context, prevalence is the proportion of people in the population who have the disease (or condition, etc). 
# Measure prevalence for eahc disease
def get_prevalence(y):
    """
    Compute prevalence.

    Args:
        y (np.array): ground truth, size (n_examples)
    Returns:
        prevalence (float): prevalence of positive cases
    """
    prevalence = 0.0
        
    prevalence = np.mean(y)
    
    return prevalence


# Test
print("Test case:\n")

y_test = np.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 1])
print(f'test labels: {y_test}')

print(f"computed prevalence: {get_prevalence(y_test)}")



util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence)

# - Sensitivity is the probability that the test outputs positive given that the case is actually positive.
# - Specificity is the probability that the test outputs negative given that the case is actually negative. 
# Calculate sensitivity and specificity for THE model:
def get_sensitivity(y, pred, th=0.5):
    """
    Compute sensitivity of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        sensitivity (float): probability that our test outputs positive given that the case is actually positive
    """
    sensitivity = 0.0
    
    # get TP and FN using our previously defined functions
    TP = true_positives(y, pred, th)
    FN = false_negatives(y, pred, th)

    # use TP and FN to compute sensitivity
    sensitivity = TP / (TP + FN)
    
    return sensitivity

def get_specificity(y, pred, th=0.5):
    """
    Compute specificity of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        specificity (float): probability that the test outputs negative given that the case is actually negative
    """
    specificity = 0.0
    
    # get TN and FP using our previously defined functions
    TN = true_negatives(y, pred, th)
    FP = false_positives(y, pred, th)
    
    # use TN and FP to compute specificity 
    specificity = TN / (TN + FP)
    
    return specificity

# Test
print("Test case")

y_test = np.array([1, 0, 0, 1, 1])
print(f'test labels: {y_test}\n')

preds_test = np.array([0.8, 0.8, 0.4, 0.6, 0.3])
print(f'test predictions: {preds_test}\n')

threshold = 0.5
print(f"threshold: {threshold}\n")

print(f"computed sensitivity: {get_sensitivity(y_test, preds_test, threshold):.2f}")
print(f"computed specificity: {get_specificity(y_test, preds_test, threshold):.2f}")


util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity)


# - Positive predictive value (PPV) is the probability that subjects with a positive screening test truly have the disease.
# - Negative predictive value (NPV) is the probability that subjects with a negative screening test truly don't have the disease.
# Calculate PPV & NPV for the model:
def get_ppv(y, pred, th=0.5):
    """
    Compute PPV of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        PPV (float): positive predictive value of predictions at threshold
    """
    PPV = 0.0
    
    # get TP and FP using our previously defined functions
    TP = true_positives(y, pred, th)
    FP = false_positives(y, pred, th)

    # use TP and FP to compute PPV
    PPV = TP / (TP + FP)
    
    return PPV

def get_npv(y, pred, th=0.5):
    """
    Compute NPV of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        NPV (float): negative predictive value of predictions at threshold
    """
    NPV = 0.0
    
    # get TN and FN using our previously defined functions
    TN = true_negatives(y, pred, th)
    FN = false_negatives(y, pred, th)

    # use TN and FN to compute NPV
    NPV = TN / (TN + FN)
    
    return NPV

# Test
print("Test case:\n")

y_test = np.array([1, 0, 0, 1, 1])
print(f'test labels: {y_test}')

preds_test = np.array([0.8, 0.8, 0.4, 0.6, 0.3])
print(f'test predictions: {preds_test}\n')

threshold = 0.5
print(f"threshold: {threshold}\n")

print(f"computed ppv: {get_ppv(y_test, preds_test, threshold):.2f}")
print(f"computed npv: {get_npv(y_test, preds_test, threshold):.2f}")


# Summarizes the model output across all thresholds, and provides a good sense of the discriminative power of a given model.
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv)


# The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. 
# Look at this curve for our model:
util.get_curve(y, pred, class_labels)
 

# Use the `sklearn` metric function of `roc_auc_score` to add this score to our metrics table.
from sklearn.metrics import roc_auc_score # a measure of goodness of fit
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv, auc=roc_auc_score)


# Create bootstrap samples and compute sample AUCs from those samples.
def bootstrap_auc(y, pred, classes, bootstraps = 100, fold_size = 1000):
    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        df.loc[:, 'y'] = y[:, c]
        df.loc[:, 'pred'] = pred[:, c]
        # get positive examples for stratified sampling
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            # stratified sampling of positive and negative examples
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)
            # to make sure that the members of each class are present
            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics

statistics = bootstrap_auc(y, pred, class_labels)


# Compute confidence intervals from the sample statistics computed.
util.print_confidence_intervals(class_labels, statistics)

# Run the following cell to generate a Precision-Recall (PRC):
util.get_curve(y, pred, class_labels, curve='prc') # useful measure of success of prediction when the classes are very imbalanced


# Use `sklearn`'s utility metric function of `f1_score` to add this measure to our performance table.
from sklearn.metrics import f1_score
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv, auc=roc_auc_score,f1=f1_score)


# Use`sklearn` library of utility `calibration_curve` to generate a calibration plot. 
from sklearn.calibration import calibration_curve
def plot_calibration_curve(y, pred): # bucketize the pridiction to a fixed number of separate bins between 0 and 1
    plt.figure(figsize=(20, 20))
    for i in range(len(class_labels)):
        plt.subplot(4, 4, i + 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(y[:,i], pred[:,i], n_bins=20)
        plt.plot([0, 1], [0, 1], linestyle='--') 
        plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
        plt.xlabel("Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title(class_labels[i])
    plt.tight_layout()
    plt.show()


# 
plot_calibration_curve(y, pred)

 
# There is a very useful method called Platt Scaling to our model's scores to resemble a well calibrated plow.
# To build this model,use the training portion of thedataset to generate the linear model.
# Then, use the model to calibrate the predictions for our test portion.
from sklearn.linear_model import LogisticRegression as LR 

y_train = train_results[class_labels].values
pred_train = train_results[pred_labels].values
pred_calibrated = np.zeros_like(pred)

for i in range(len(class_labels)):
    lr = LR(solver='liblinear', max_iter=10000)
    lr.fit(pred_train[:, i].reshape(-1, 1), y_train[:, i])    
    pred_calibrated[:, i] = lr.predict_proba(pred[:, i].reshape(-1, 1))[:,1]


plot_calibration_curve(y[:,], pred_calibrated)

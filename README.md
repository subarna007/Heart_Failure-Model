# Heart failure survival prediction

Predicting death during follow-up for patients with heart failure from 12 clinical features, with logistic regression and k-nearest neighbours.

## Data

`heartfailure.csv` is the Heart Failure Clinical Records dataset (299 patients, 96 deaths), available from the UCI Machine Learning Repository and Kaggle. Features include age, ejection fraction, serum creatinine, serum sodium, anaemia, diabetes, high blood pressure, platelets, sex, smoking and follow-up time. The target is `DEATH_EVENT`.

## Notebooks

- `Heart Failure Prediction.ipynb`: exploration, correlation analysis, logistic regression with 5-fold cross-validated regularisation, and a k-NN model on standardised features
- `KNN.ipynb`: k-NN without scaling, testing k from 1 to 49 and plotting train and test accuracy against k

## Results

80/20 train/test split, 60 test patients.

| Model | Test accuracy | Sensitivity for death | F1 for death |
|---|---|---|---|
| Logistic regression (`random_state=1`) | 88% | 64% (9 of 14) | 0.72 |
| k-NN, k = 5, standardised features (`random_state=42`) | 68% | 28% (7 of 25) | 0.42 |

The k-NN sweep in `KNN.ipynb` reached its highest test accuracy, 76.7%, at k = 26. Because k was chosen by looking at the test set, that number is optimistic.

## What I would do differently

- Drop the `time` feature: it is the length of follow-up, which is only known afterwards, so using it leaks the outcome into the model.
- Choose k and other settings with cross-validation on the training set only.
- Use one fixed split for all models, so their results can be compared directly.
- With only 299 patients, report cross-validated results with confidence intervals rather than a single split.

## Run

```bash
pip install -r requirements.txt
jupyter notebook
```

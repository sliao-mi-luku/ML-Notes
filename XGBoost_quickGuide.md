# XGBoost - Quck Quide of Implementation

*Last Updated: 2023-02-13*


**References:**

- https://xgboost.readthedocs.io/en/stable/get_started.html
- https://xgboost.readthedocs.io/en/stable/python/python_api.html
- https://www.udacity.com/course/aws-machine-learning-engineer-nanodegree--nd189


#### Quick Implementation

```python
import xgboost as xgb

# convert Pandas DataFrame into DMatrix
df_dmatrix = xgb.DMatrix(df, label=df['y'])

# create model and train
model = xgb.train(params={"eval_metric": ..., "objective": ...}, dtrain=df_dmatrix)

# inference
model.predict(df_dmatrix)
```

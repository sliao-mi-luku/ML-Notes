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


#### With SageMaker

```python
import sagemaker

session = sagemaker.Session()

container = sagemaker.image_uris.retrieve('xgboost', session.boto_region_name, version='latest')

xgb = sagemaker.estimator.Estimator(container, role, instance_count=1, instance_type='ml.m5.xlarge',
                                    output_path="{}/output".format(session.default_bucket()),
                                    sagemaker_session=session)

xgb.set_hyperparameters(max_depth=5, eta=0.2, gamma=4, min_child_weight=6, subsample=0.8, objective='reg:linear', early_stopping_rounds=10, num_round=200)

train_data = sagemaker.inputs.TrainingInput(s3_data='<TRAIN_PATH>', content_type='csv')

val_data = sagemaker.inputs.TrainingInput(s3_data='<VAL_PATH>', content_type='csv')

xgb.fit({'train': train_data, 'validation': val_data})
```

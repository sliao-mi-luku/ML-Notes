# AutlGluon - Quck Quide of Implementation

*Last Updated: 2023-02-13*


**References:**

- https://auto.gluon.ai/stable/index.html
- https://www.udacity.com/course/aws-machine-learning-engineer-nanodegree--nd189


#### Quick Implementation

```python
from autogluon.tabular import TabularPredictor

# create model and train
# classification
model = TabularPredictor(label='y').fit(train_data=df, time_limit=100, presets='best_quality')
# regression
model = TabularPredictor(label='y', problem_type='regression', eval_metric='r2').fit(train_data=df, time_limit=100, presets='best_quality')


model.fit_summary()

# compare models
model.leaderboard()

model.leaderboard(silent=True).plot(kind="bar", x="model", y="score_val")

# inference
model.evaluate(df_test)
```

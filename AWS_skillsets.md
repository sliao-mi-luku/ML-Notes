# Machine Learning Workflows with AWS


## Data Wrangling

- Feature store

## Model Training

- Available images ([lists](https://github.com/aws/deep-learning-containers/blob/master/available_images.md))
- TensorFlow estimator ([Example codes](https://github.com/aws/amazon-sagemaker-examples/tree/main/frameworks/tensorflow))
- PyTorch estimator ([Example codes](https://github.com/aws/amazon-sagemaker-examples/tree/main/frameworks/pytorch))
- Bring your own model ([Example codes](https://github.com/aws/amazon-sagemaker-examples/tree/main/advanced_functionality))([AWS page](https://aws.amazon.com/blogs/machine-learning/bring-your-own-model-with-amazon-sagemaker-script-mode/))
- Hyperparameter tuning ([Example codes](https://github.com/aws/amazon-sagemaker-examples/tree/main/hyperparameter_tuning))
- SageMaker debugger/profiling ([Example codes](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-debugger))

## Model Deployment

- Endpoint ([AWS page](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html))
- Multi-AZ deployment ([AWS page](https://aws.amazon.com/blogs/database/amazon-rds-under-the-hood-multi-az/))
- Batch transform ([AWS page](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html))
- CreateModel API

## Model Monitoring

- Model monitor ([AWS page](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html))
- Clarify ([AWS page](https://aws.amazon.com/sagemaker/clarify/?sagemaker-data-wrangler-whats-new.sort-by=item.additionalFields.postDateTime&sagemaker-data-wrangler-whats-new.sort-order=desc))
- Model retraining

## Workflow Design

- AWS Lambda ([AWS page](https://docs.aws.amazon.com/lambda/latest/dg/welcome.html))
- AWS Step Function

## Model Operationalizing

- Spot instances (EC2) ([AWS page](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html))
- Distributed computing ([AWS page](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html))
- Distributed data (EC2) ([parameters](https://docs.aws.amazon.com/cdk/api/v1/python/aws_cdk.aws_stepfunctions_tasks/S3DataDistributionType.html))
- Data stores ([Reviews](https://www.missioncloud.com/blog/resource-amazon-ebs-vs-efs-vs-s3-picking-the-best-aws-storage-option-for-your-business))
- Lambda function
- Automatic scaling / Concurrency ([AWS page](https://docs.aws.amazon.com/lambda/latest/operatorguide/scaling-concurrency.html))
- Feature store
- IAM
- Virtual Private Cloud (VPC)

## References

- Data Science on AWS (O'Reilly) https://learning.oreilly.com/library/view/data-science-on/9781492079385/
- Udacity https://www.udacity.com/course/aws-machine-learning-engineer-nanodegree--nd189
- Udacity's GitHub https://github.com/udacity/udacity-nd009t-C2-Developing-ML-Workflow

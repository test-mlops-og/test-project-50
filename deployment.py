# deployment_script.py
# Auto-generated from deployment.ipynb

import yaml
from sagemaker.serve import SchemaBuilder
from sagemaker.serve import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
import mlflow
from mlflow import MlflowClient
import numpy as np
import os

# Configuration
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    config_file = os.getenv("CONFIG_PATH", "config.yaml")
    config = load_config(config_file)
    # Extract configuration values from YAML
    tracking_server_arn = config.get("tracking_server_arn")
    role = config.get("role")
    model_package_group_name = config.get("model_package_group_name")

    # Setup MLflow tracking server URI using the ARN from the config
    mlflow.set_tracking_uri(tracking_server_arn)
    client = MlflowClient()

    # Get the registered model using model_package_group_name from the config
    registered_model = client.get_registered_model(name=model_package_group_name)
    source_path = registered_model.latest_versions[0].source

    # Example input data for the sklearn model
    sklearn_input = np.array([
          -0.6161975481284616,
          -0.6840938444613145,
          -0.4666533461032037,
          -0.6857462206781514,
          -0.5333397337170492,
          -0.6669056147788445,
          -0.8537557997803805,
          0,
          1,
          0
        ]).reshape(1, -1)
    sklearn_output = 1

    # Build the schema using the sample input and output
    sklearn_schema_builder = SchemaBuilder(
        sample_input=sklearn_input,
        sample_output=sklearn_output,
    )

    # Retrieve deployment configuration
    deployment_config = config.get("deployment", {})
    environments = deployment_config.get("environments", [])

    # Loop over each deployment environment
    for env in environments:
        print(f"Deploying for environment: {env}")
        model_name = f"{model_package_group_name}-{env}"

        # Create model builder with the environment-specific name and metadata.
        model_builder = ModelBuilder(
            name=model_name,
            mode=Mode.SAGEMAKER_ENDPOINT,
            schema_builder=sklearn_schema_builder,
            role_arn=role,
            model_metadata={"MLFLOW_MODEL_PATH": source_path},
        )

        built_model = model_builder.build()

        # Use instance type from deployment configuration (default if not provided)
        instance_type = deployment_config.get("instance_type", "ml.m5.large")
        predictor = built_model.deploy(initial_instance_count=1, instance_type=instance_type)

        # Run a prediction using the deployed endpoint
        result = predictor.predict(sklearn_input)
        print(f"Prediction result for {env} environment:", result)

if __name__ == '__main__':
    main()

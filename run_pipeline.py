# run_pipeline.py

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.sweep import Choice
from azure.ai.ml.dsl import pipeline

step_process = command(
    name="data_preparation",
    display_name="Data Preparation for Automated Vehicle Pricing",
    description="Prepare and split data into train and test sets",
    inputs={
        "data": Input(type="uri_file"),
        "test_train_ratio": Input(type="number"),
    },
    outputs={
        "train_data": Output(type="uri_folder", mode="rw_mount"),
        "test_data": Output(type="uri_folder", mode="rw_mount"),
    },
    
    code="src/data_prep", 
    command="""python data_prep.py \
            --data ${{inputs.data}} \
            --test_train_ratio ${{inputs.test_train_ratio}} \
            --train_data ${{outputs.train_data}} \
            --test_data ${{outputs.test_data}}""",
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
    compute="cpu-cluster",
)

train_step = command(
    name="train_price_prediction_model",
    display_name="Train Price Prediction Model",
    description="Train a Random Forest Regressor for used car price prediction",
    inputs={
        "train_data": Input(type="uri_folder"),
        "test_data": Input(type="uri_folder"),
        "n_estimators": Input(type="number", default=100),
        "max_depth": Input(type="number", default=10),
    },
    outputs={
        "model_output": Output(type="mlflow_model"),
    },
    code="src/model_train",
    command="""python model_train.py \
            --train_data ${{inputs.train_data}} \
            --test_data ${{inputs.test_data}} \
            --n_estimators ${{inputs.n_estimators}} \
            --max_depth ${{inputs.max_depth}} \
            --model_output ${{outputs.model_output}}""",
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
    compute="cpu-cluster",
)

model_register_component = command(
    name="register_model",
    display_name="Register Best Model",
    description="Register the best trained model in MLflow Model Registry",
    inputs={
        "model": Input(type="mlflow_model"),
    },
    code="src/model_register",
    command=
    """python  model_register.py \
            --model ${{inputs.model}}""",
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
    compute="cpu-cluster",
)

# -----------------------------
# PIPELINE
# -----------------------------

@pipeline(
    compute="cpu-cluster",
    description="End-to-end MLOps pipeline for used car price prediction",
)
def complete_pipeline(input_data_uri, test_train_ratio, n_estimators, max_depth):

    # Step 1: Preprocess the data
    preprocess_step = step_process(
        data=input_data_uri,
        test_train_ratio=test_train_ratio,
    )

    # Step 2: Train the model using preprocessed data (sweep over hyperparams)
    job_for_sweep = train_step(
        train_data=preprocess_step.outputs.train_data,
        test_data=preprocess_step.outputs.test_data,
        n_estimators=Choice(values=[10, 20, 30, 50]),
        max_depth=Choice(values=[5, 10, 15, 20]),
    )

    sweep_job = job_for_sweep.sweep(
        compute="cpu-cluster",
        sampling_algorithm="random",
        primary_metric="MSE",
        goal="Minimize",
    )

    sweep_job.set_limits(max_total_trials=20, max_concurrent_trials=10, timeout=7200)

    # Step 3: Register the best model
    model_register_step = model_register_component(
        model=sweep_job.outputs.model_output,
    )

    return {
        "pipeline_job_train_data": preprocess_step.outputs.train_data,
        "pipeline_job_test_data": preprocess_step.outputs.test_data,
        "pipeline_job_best_model": job_for_sweep.outputs.model_output,
    }

# -----------------------------
# MAIN: submit the pipeline job
# -----------------------------

def get_ml_client():
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception:
        credential = InteractiveBrowserCredential()

    # ml_client = MLClient.from_config(
    #     credential=credential,
    #     path="workspace.json",  
    # )
    ml_client = MLClient.from_config(credential=credential)

    return ml_client


def main():
    ml_client = get_ml_client()

    DATA_ASSET_NAME = "used-cars-data"
    DATA_VERSION = "9"  

    data_path = ml_client.data.get(DATA_ASSET_NAME, version=DATA_VERSION).path

    pipeline_instance = complete_pipeline(
        input_data_uri=Input(type="uri_file", path=data_path),
        test_train_ratio=0.2,
        n_estimators=50,
        max_depth=5,
    )

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_instance,
        experiment_name="price_prediction_pipeline",
    )

    print(f"Pipeline submitted: {pipeline_job.name}")
    if pipeline_job.services and "Studio" in pipeline_job.services:
        print(f"Web View: {pipeline_job.services['Studio'].endpoint}")


if __name__ == "__main__":
    main()

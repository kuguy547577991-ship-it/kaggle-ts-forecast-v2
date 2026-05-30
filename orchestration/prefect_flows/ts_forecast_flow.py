"""Prefect Flow for TS forecast training."""

from prefect import flow, task


@task
def train_ts_forecast():
    import subprocess
    result = subprocess.run(
        ["py", "pipelines/ts_forecast/pipeline.py", "--config", "local"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed: {result.stderr}")
    return result.stdout


@flow(name="ts-forecast-training")
def ts_forecast_flow():
    output = train_ts_forecast()
    print(output)


if __name__ == "__main__":
    ts_forecast_flow()

import mlflow
from ssl_tools.pipelines.base import Pipeline, auto_main


def EvaluateAll(Pipeline):
    def __init__(
        self, 
        experiment_id: str,
        experiment_name: str,
        log_dir: str = "./mlruns",
        skip_existing: bool = True,
    ):
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.skip_existing = skip_existing
        self.log_dir = log_dir
        
    def run(self):
        client = mlflow.client.MlflowClient(tracking_uri=self.log_dir)

if __name__ == "__main__":
    options = {"all": EvaluateAll}
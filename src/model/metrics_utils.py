import json

def save_metrics(metrics: dict, file_path: str) -> None:
    """
    Save evaluation metrics to a JSON file.
    Args:
        metrics (dict): Dictionary of metrics to save.
        file_path (str): Path to the output JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)

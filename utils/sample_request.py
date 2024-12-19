import base64
import os
import pathlib
import json
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict

PROJECT = "ck-vertex"
LOCATION = "us-central1"
ENDPOINT_ID = "6057421763162144768"
PATH_TO_IMG = pathlib.Path(__file__).parent.absolute()

def predict_image_classification_sample(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    with open(filename, "rb") as f:
        file_content = f.read()

    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]

    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5,
        max_predictions=5,
    ).to_value()

    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )

    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )

    # Convert to a regular Python dict (dealing with protobufs here)
    predictions = [dict(prediction) for prediction in response.predictions]

    for pred in predictions:
        pred['confidences'] = list(pred['confidences'])
        pred['displayNames'] = list(pred['displayNames'])
        pred['ids'] = list(pred['ids'])

    return predictions

if __name__ == "__main__":
    predictions = predict_image_classification_sample(
        filename=PATH_TO_IMG,
        project=PROJECT,
        location=LOCATION,
        endpoint_id=ENDPOINT_ID
    )


    print(json.dumps(predictions, indent=2))
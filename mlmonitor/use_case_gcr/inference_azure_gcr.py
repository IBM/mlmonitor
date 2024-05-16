# SPDX-License-Identifier: Apache-2.0
import json
import pandas as pd
import joblib
import os


def init():
    # TODO pick model name from Container environment variables set to "model.joblib"
    global model
    print(f"loading model form {os.getenv('AZUREML_MODEL_DIR')}")
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.joblib")
    print(model_path)
    model = joblib.load(model_path)


def run(input_data):
    print("start inference")
    try:
        if type(input_data) is str:
            dict_data = json.loads(input_data)
            print(f"input data (str):\n{dict_data}")
        else:
            dict_data = input_data
            print(f"input data (json):\n{dict_data}")

        data = pd.DataFrame.from_dict(dict_data["input"])
        print(data)
        predictions = model.predict(data)
        print(predictions)
        scores = model.predict_proba(data).tolist()
        records = [
            {"Scored Labels": int(pred), "Scored Probabilities": prob}
            for pred, prob in zip(predictions, scores)
        ]
        result = {"output": records}
        print(f"output:data:\n{result}")

        return result
    except Exception as e:
        result = str(e)
        # return error message back to the client
        print(f"output:error:\n{result}")
        return {"error": result}

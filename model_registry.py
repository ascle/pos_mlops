
# ## import das bibliotecas
import os
import mlflow
import numpy as np

# ## Configurando o MLflow

MLFLOW_TRACKING_URI = 'https://dagshub.com/ascle/pos_mlops.mlflow'
MLFLOW_TRACKING_USERNAME = 'ascle'
MLFLOW_TRACKING_PASSWORD = '4e7865ca4e100c3a50288fe551f551c83f08a298'
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ## Criando um client para comunicar com o registro no DagsHub

client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

print(client)

# ## Recebendo o modelo registrado e suas versões
registered_model = client.get_registered_model('predito')
registered_model.latest_versions

# ## Obtendo o id da execução do modelo
run_id = registered_model.latest_versions[-1].run_id

# ## Carregando o modelo
logged_model = f'runs:/{run_id}/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

# ## Fazendo uma predição com o modelo carregadO
accelerations = 0
fetal_movement = 0
uterine_contractions = 0
severe_decelerations = 0

received_data = np.array([
        accelerations,
        fetal_movement,
        uterine_contractions,
        severe_decelerations,
    ]).reshape(1, -1)
received_data
loaded_model.predict(received_data)

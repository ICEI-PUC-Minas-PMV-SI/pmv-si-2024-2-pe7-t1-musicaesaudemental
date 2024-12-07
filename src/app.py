from pathlib import Path
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import json
from collections import Counter
from sklearn.metrics import precision_score, accuracy_score
from xgboost import Booster

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
booster = Booster()

# Carregar os modelos treinados
models = {
    "decision-tree": joblib.load("src/models/decision-tree-classifier.pkl"),
    "decision-tree-undersampling": joblib.load("src/models/decision-tree-classifier-undersampling.pkl"),
    "decision-tree-oversampling": joblib.load("src/models/decision-tree-classifier-oversampling.pkl"),
    "decision-tree-smote": joblib.load("src/models/decision-tree-classifier-smote.pkl"),
    "decision-tree-xgboost": joblib.load("src/models/decision-tree-classifier-xgboost.pkl")
}

# Carregar métricas individualmente
model_metrics = {
    "decision-tree": json.load(open("src/metrics/decision-tree-classifier.json")),
    "decision-tree-undersampling": json.load(open("src/metrics/decision-tree-classifier-undersampling.json")),
    "decision-tree-oversampling": json.load(open("src/metrics/decision-tree-classifier-oversampling.json")),
    "decision-tree-smote": json.load(open("src/metrics/decision-tree-classifier-smote.json")),
    "decision-tree-xgboost": json.load(open("src/metrics/decision-tree-classifier-xgboost.json"))
}

effect_mapping = {0: 'Worsen', 1: 'No effect', 2: 'Improve'}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running"}), 200

@app.route('/predict-all', methods=['POST'])
def predict_all():
    try:
        # Receber dados do payload
        input_data = request.get_json()

        # Verificar se é uma lista de dicionários
        if not isinstance(input_data, list) or len(input_data) == 0:
            return jsonify({"status": "error", "message": "Payload deve ser uma lista de objetos JSON"}), 400

        # Converter para DataFrame
        df = pd.DataFrame(input_data)

        results = []

        for model_name, model in models.items():
            try:
                # Fazer a previsão
                y_pred = model.predict(df)
                predicted_effect = effect_mapping[y_pred[0]]

                # Recuperar métricas do arquivo JSON
                metrics = model_metrics[model_name]
                precision_per_class = metrics.get("precision_per_class", {})
                overall_precision = metrics.get("precision", 0)
                accuracy = metrics.get("accuracy", 0)

                # Adicionar resultados
                results.append({
                    "accuracy": accuracy,
                    "precision": overall_precision,
                    "precision_per_class": precision_per_class,
                    "model": model_name,
                    "musicEffect": predicted_effect
                })
            except Exception as e:
                results.append({
                    "accuracy": None,
                    "precision": None,
                    "model": model_name,
                    "musicEffect": f"Erro: {str(e)}"
                })
        print(results)
        if results:
            valid_results = [r for r in results if r["accuracy"] is not None and r["precision"] is not None]
            if not valid_results:
                return jsonify({"status": "error", "message": "Nenhum resultado válido"}), 400

            # Contar previsões de classes
            class_counts = Counter(r["musicEffect"] for r in valid_results)
            most_common_class, max_count = class_counts.most_common(1)[0]
            print(class_counts)
            # Filtrar modelos que retornaram a classe mais comum
            common_class_candidates = [
                r for r in valid_results if r["musicEffect"] == most_common_class
            ]
            print(common_class_candidates)

            # Resolver empate por precisão geral
            if len(common_class_candidates) > 1:
                max_precision = max(r["precision"] for r in common_class_candidates)
                common_class_candidates = [
                    r for r in common_class_candidates if r["precision"] == max_precision
                ]

            # Resolver empate por acurácia
            if len(common_class_candidates) > 1:
                max_accuracy = max(r["accuracy"] for r in common_class_candidates)
                common_class_candidates = [
                    r for r in common_class_candidates if r["accuracy"] == max_accuracy
                ]

            # Caso ainda haja empate, retorno todos os candidatos
            if len(common_class_candidates) > 1:
                return jsonify({"status": "empate", "candidates": common_class_candidates}), 200

            # Melhor modelo escolhido
            best_model = common_class_candidates[0]
            response = {
                "status": "success",
                "best_model": best_model,
                "metrics": {
                    "overall_precision": best_model["precision"],
                    "class_precision": best_model["precision_per_class"].get(
                        str(next(k for k, v in effect_mapping.items() if v == best_model["musicEffect"])), 0
                    )
                }
            }
            return jsonify(response), 200

        return jsonify({"status": "error", "message": "Nenhum resultado foi processado"}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
 
   

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

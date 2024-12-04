from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.metrics import precision_score, accuracy_score
from xgboost import Booster

app = Flask(__name__)
booster = Booster()

# Carregar os modelos treinados
models = {
    "decision-tree": joblib.load("src/models/decision-tree-classifier.pkl"),
    "decision-tree-undersampling": joblib.load("src/models/decision-tree-classifier-undersampling.pkl"),
    "decision-tree-oversampling": joblib.load("src/models/decision-tree-classifier-oversampling.pkl"),
    "decision-tree-smote": joblib.load("src/models/decision-tree-classifier-smote.pkl"),
    "decision-tree-xgboost": joblib.load("src/models/decision-tree-classifier-xgboost.pkl")
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

        # Verificar se "labels" está presente no DataFrame
        if 'labels' in df.columns:
            y_test = df.pop('labels')  # Retirar os rótulos para calcular métricas
        else:
            return jsonify({"status": "error", "message": "Payload deve conter a coluna 'labels' com os rótulos reais"}), 400

        # Garantir que y_test seja numérico
        if isinstance(y_test.iloc[0], str):
            reverse_mapping = {v: k for k, v in effect_mapping.items()}
            y_test = y_test.map(reverse_mapping)

        results = []

        for model_name, model in models.items():
            try:
                # Fazer a previsão
                y_pred = model.predict(df)

                # Calcular métricas
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)

                # Converter previsões para rótulos descritivos
                y_pred_labels = [effect_mapping[pred] for pred in y_pred]

                # Adicionar resultados
                results.append({
                    "accuracy": round(accuracy, 2),
                    "precision": round(precision, 2),
                    "model": model_name,
                    "musicEffect": y_pred_labels[0] if y_pred_labels else "Unknown"
                })
            except Exception as e:
                results.append({
                    "accuracy": None,
                    "precision": None,
                    "model": model_name,
                    "musicEffect": f"Erro: {str(e)}"
                })

        if results:
            valid_results = [r for r in results if r["accuracy"] is not None and r["precision"] is not None]
            
            if not valid_results:
                return jsonify({"status": "error", "message": "Nenhum resultado válido"}), 400
            
            max_precision = max(r["precision"] for r in valid_results)
            candidates = [r for r in valid_results if r["precision"] == max_precision]

            
            if len(candidates) > 1:
                max_accuracy = max(r["accuracy"] for r in candidates)
                candidates = [r for r in candidates if r["accuracy"] == max_accuracy]

            
            if len(candidates) > 1:
                effect_counts = {}
                for r in candidates:
                    effect = r["musicEffect"]
                    effect_counts[effect] = effect_counts.get(effect, 0) + 1
                most_frequent_effect = max(effect_counts, key=effect_counts.get)
                candidates = [r for r in candidates if r["musicEffect"] == most_frequent_effect]

            if len(candidates) > 1:
                return jsonify({"status": "empate", "candidates": candidates}), 200
            
            best_model = candidates[0]
            return jsonify({"status": "success", "best_model": best_model}), 200

        return jsonify({"status": "error", "message": "Nenhum resultado foi processado"}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
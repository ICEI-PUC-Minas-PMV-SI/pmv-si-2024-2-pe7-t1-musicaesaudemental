# CASO 1 - Árvore de Decisão

# Preparação dos dados
Para a prepação dos dados que serão utilizados no modelo, utilizamos os passos a seguir:

## Limpeza dos dados
Realizamos a limpeza dos valores ausentes na coluna 'Music effects'

```
data_clean = data.dropna(subset=['Music effects'])
```

## Codificacao dos dados
Realizamos o mapeamento da variável alvo 'Music effects' para incluir o 'Worsen" e codificando os valores 'Improve', 'No effect' e 'Worsen' para 1, 0 e -1, respectivamente.

```
data_clean['Music effects'] = data_clean['Music effects'].map({'Improve': 1, 'No effect': 0, 'Worsen': -1})
```

## Variáveis consideradas para treinamento do modelo
Para realizar o treinamento e teste do modelo, considerando o que esperamos no nosso output, utilizamos as variáveis X e y, onde X contém as colunas de frequência dos gêneros, colunas das doenças mentais e a idade e y contém os efeitos da musica no tratamento.

```
X = data[style_columns + ['Anxiety', 'Depression', 'Insomnia', 'OCD', 'Age']]
y = data['Music effects']
```

## Separação dos dados 
Para o treinamento e teste do modelo, separamos os dados em 70% do dataset para treinamento e 30% para testes do modelo.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

# Descrição dos modelos
## Árvore de Decisão
O algoritmo Arvore de Decisão(Decision Tree) foi escolhido, pois é uma abordagem simples, eficiente e intuitiva para modelar o impacto da música na saúde mental, oferecendo boa interpretabilidade e capacidade de capturar padrões complexos sem a necessidade de suposições rígidas ou tratamentos iniciais elaborados. Elas representam um bom ponto de partida para explorar os dados e compreender as principais variáveis que influenciam os efeitos da música.
Alguns dos principais motivos para a escolha por árvore de descisão:
1. Interpretação clara: Árvores de decisão são fáceis de entender e interpretar. Cada divisão na árvore representa uma regra baseada nos dados, permitindo identificar como diferentes variáveis influenciam os resultados.

2. Lida com dados heterogêneos: Elas podem trabalhar com variáveis contínuas e categóricas sem a necessidade de transformações complexas, o que é especialmente útil neste problema, onde os dados incluem frequências de escuta de música e condições psicológicas.

3. Identificação de padrões relevantes: Árvores de decisão podem capturar relações não-lineares e interações entre variáveis de forma automática. Isso é importante em um problema como este, onde os efeitos da música podem depender de combinações de fatores (por exemplo, um gênero específico impactar mais em casos de ansiedade elevada).

4.Flexibilidade: Árvores de decisão funcionam bem com problemas de classificação multiclasses, como este, sem exigir alterações substanciais nos dados ou na modelagem.

5. Não requer suposições: Diferentemente de modelos estatísticos como a regressão logística, árvores de decisão não fazem suposições sobre a distribuição dos dados ou a linearidade das relações, o que as torna mais versáteis para problemas com relações complexas entre variáveis.

Para a construção do modelo, foi realizado a importação das seguintes bibliotecas e o dataset foi carregado em memória pelo pandas.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = 'C:/Users/tulio/Downloads/dados_tratados_V2.csv'
data = pd.read_csv(file_path, delimiter=';')
```

No trecho de código a seguir, fazemos o carregamento de algumas informações do dataset

```
print(data.head())
print(data.info())
```

O trecho a seguir é responsável por armazenar a frequencia dos estilos musicais e a relação entre frequência dos gêneros musicais e o efeitos no tratamento
```
style_columns = [col for col in data.columns if 'frequency' in col]
data[style_columns].hist(bins=15, figsize=(15, 10))
plt.show()

# Relação entre frequência de gêneros musicais e "Music effects"
sns.countplot(x='Music effects', data=data)
plt.show()
```

Aqui é feito a limpeza e codificação dos dados e também é definido as variáveis X e y
```
data_clean = data.dropna(subset=['Music effects'])
data_clean['Music effects'] = data_clean['Music effects'].map({'Improve': 1, 'No effect': 0, 'Worsen': -1})

X = data[style_columns + ['Anxiety', 'Depression', 'Insomnia', 'OCD', 'Age']]
y = data['Music effects']
```

Neste trecho é feito a separação dos dados para teste e treinamento
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Fazemos a definição de parâmetros que serão utilizados na criação do modelo

```
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}
```

Após a aplicação da limpeza, codificação dos dados, definição das variáveis e parâmetros seguimos para a criação do modelo

```
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```
Após a predição do modelo, segue para o cálculo das métricas de qualidade e os melhores parâmtetros

```
y_pred = best_model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Melhores parâmetros:", grid_search.best_params_)
```
Após a análise das métricas de qualidade, é feito a visualização da árvore de decisão

```
plt.figure(figsize=(25, 15))
plot_tree(best_model, feature_names=X.columns, class_names=['Worsen', 'Improve', 'No effect'], filled=True)
plt.show()
```
### Árvore de Decisão com Undersampling

Para a construção do modelo, foram importadas as bibliotecas necessárias e o dataset foi carregado em memória usando o pandas.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler  # Undersampling

file_path = 'C:/Users/tulio/Downloads/dados_tratados_V2.csv'
data = pd.read_csv(file_path, delimiter=';')
```

O trecho de código a seguir é responsável por carregar algumas informações básicas do dataset

```
print(data.head())
print(data.info())
```

Neste trecho, são exploradas a distribuição dos estilos musicais e a relação entre a frequência de gêneros musicais e os efeitos percebidos.
```
style_columns = [col for col in data.columns if 'frequency' in col]
data[style_columns].hist(bins=15, figsize=(15, 10))
plt.show()

sns.countplot(x='Music effects', data=data)
plt.show()
```

Aqui é feito a limpeza e codificação dos dados e também é definido as variáveis X e y
```
data_clean = data.dropna(subset=['Music effects'])

data_clean['Music effects'] = data_clean['Music effects'].map({'Improve': 1, 'No effect': 0, 'Worsen': -1})

X = data[style_columns + ['Anxiety', 'Depression', 'Insomnia', 'OCD', 'Age']]
y = data['Music effects']
```

Como as classes estavam desbalanceadas, utilizamos RandomUnderSampler para realizar um balanceamento das amostras.
```
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
```

Neste trecho é feito a separação dos dados para teste e treinamento
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Fazemos a definição de parâmetros que serão utilizados na criação do modelo
```
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}
```

Após a aplicação da limpeza, codificação dos dados, definição das variáveis e parâmetros seguimos para a criação do modelo
```
model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Melhores parâmetros:", grid_search.best_params_)
```

Após a previsão, métricas de qualidade, como acurácia e relatório de classificação, foram calculadas. Também foi gerada a matriz de confusão para visualizar a performance do modelo.
```
y_pred = best_model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Worsen', 'No effect', 'Improve'], yticklabels=['Worsen', 'No effect', 'Improve'])
plt.title("Matriz de Confusão")
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.show()
```

Por fim, a estrutura da árvore de decisão gerada foi visualizada para interpretar as regras criadas pelo modelo.
```
plt.figure(figsize=(25, 15))
plot_tree(best_model, feature_names=X.columns, class_names=['Worsen', 'Improve', 'No effect'], filled=True)
plt.show()
```

### Árvore de Decisão com Oversampling

Para a construção do modelo, foram importadas as bibliotecas necessárias e o dataset foi carregado em memória usando o pandas.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading dataset
file_path = 'C:/Users/tulio/Downloads/dados_tratados_V2.csv'
data = pd.read_csv(file_path, delimiter=';')
```

O trecho de código a seguir é responsável por carregar algumas informações básicas do dataset

```
print(data.head())
print(data.info())
```

Neste trecho, são exploradas a distribuição dos estilos musicais e a relação entre a frequência de gêneros musicais e os efeitos percebidos.
```
style_columns = [col for col in data.columns if 'frequency' in col]
data[style_columns].hist(bins=15, figsize=(15, 10))
plt.show()

sns.countplot(x='Music effects', data=data)
plt.show()
```

Aqui é feito a limpeza e codificação dos dados e também é definido as variáveis X e y
```
data_clean = data.dropna(subset=['Music effects'])

data_clean['Music effects'] = data_clean['Music effects'].map({'Improve': 1, 'No effect': 0, 'Worsen': -1})

X = data[style_columns + ['Anxiety', 'Depression', 'Insomnia', 'OCD', 'Age']]
y = data['Music effects']
```

Os dados foram divididos em conjuntos de treino e teste, preservando a distribuição proporcional das classes (stratify)
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Foi realizada uma reamostragem para equilibrar as classes no conjunto de treino. As classes minoritárias foram replicadas até atingir o mesmo número da classe majoritária.
```
train_data = pd.concat([X_train, y_train], axis=1)
class_counts = train_data['Music effects'].value_counts()
print("Contagem original das classes:\n", class_counts)

max_class = class_counts.max()

train_oversampled = pd.concat([
    train_data[train_data['Music effects'] == cls].sample(max_class, replace=True, random_state=42)
    for cls in class_counts.index
])

print("Contagem após oversampling:\n", train_oversampled['Music effects'].value_counts())

X_train_oversampled = train_oversampled.drop(columns=['Music effects'])
y_train_oversampled = train_oversampled['Music effects']
```

Foram definidos hiperparâmetros para serem ajustados no modelo utilizando GridSearchCV
```
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}
```

O modelo foi treinado com GridSearchCV, e o melhor conjunto de parâmetros foi selecionado para realizar as previsões
```
model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_oversampled, y_train_oversampled)

best_model = grid_search.best_estimator_
print("Melhores parâmetros:", grid_search.best_params_)
```

Após realizar as previsões no conjunto de teste, foram calculadas métricas de qualidade, como acurácia e relatório de classificação. A matriz de confusão foi gerada para análise detalhada do desempenho.
```
y_pred = best_model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Worsen', 'No effect', 'Improve'], yticklabels=['Worsen', 'No effect', 'Improve'])
plt.title("Matriz de Confusão")
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.show()
```

A estrutura da árvore de decisão foi visualizada para melhor interpretar as decisões tomadas pelo modelo
```
plt.figure(figsize=(25, 15))
plot_tree(best_model, feature_names=X.columns, class_names=['Worsen', 'Improve', 'No effect'], filled=True)
plt.show()
```

### Árvore de Decisão com SMOTE

Para a construção do modelo, foram importadas as bibliotecas necessárias e o dataset foi carregado em memória usando o pandas.
```
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

file_path = 'C:/Users/tulio/Downloads/dados_tratados_V2.csv'
data = pd.read_csv(file_path, delimiter=';')
```

O trecho de código a seguir é responsável por carregar algumas informações básicas do dataset

```
print(data.head())
print(data.info())
```

Neste trecho, são exploradas a distribuição dos estilos musicais e a relação entre a frequência de gêneros musicais e os efeitos percebidos.
```
style_columns = [col for col in data.columns if 'frequency' in col]
data[style_columns].hist(bins=15, figsize=(15, 10))
plt.show()

sns.countplot(x='Music effects', data=data)
plt.show()
```

Aqui é feito a limpeza e codificação dos dados e também é definido as variáveis X e y
```
data_clean = data.dropna(subset=['Music effects'])

data_clean['Music effects'] = data_clean['Music effects'].map({'Improve': 1, 'No effect': 0, 'Worsen': -1})

X = data[style_columns + ['Anxiety', 'Depression', 'Insomnia', 'OCD', 'Age']]
y = data['Music effects']
```

Os dados foram divididos em conjuntos de treino e teste, preservando a distribuição proporcional das classes (stratify)
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

O método SMOTE foi utilizado para gerar amostras sintéticas e equilibrar as classes da variável dependente.
```
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

Foram definidos hiperparâmetros para serem ajustados no modelo utilizando GridSearchCV
```
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}
```

Utilizando GridSearchCV, foi realizado o treinamento do modelo de árvore de decisão, considerando o balanceamento das classes.
```
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='f1_macro'
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

```

O modelo foi avaliado no conjunto de teste, e as métricas de qualidade foram calculadas, incluindo a matriz de confusão.
```
y_pred = best_model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Worsen', 'No effect', 'Improve'], yticklabels=['Worsen', 'No effect', 'Improve'])
plt.title("Matriz de Confusão")
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.show()

print("Melhores parâmetros:", grid_search.best_params_)
```

A estrutura da árvore de decisão foi visualizada para melhor interpretar as decisões tomadas pelo modelo
```
plt.figure(figsize=(25, 15))
plot_tree(best_model, feature_names=X.columns, class_names=['Worsen', 'Improve', 'No effect'], filled=True)
plt.show()
```

### Árvore de Decisão com SMOTE e XGBoost

O código utiliza diversas bibliotecas para manipulação, visualização e modelagem dos dados, incluindo pandas para manipulação do dataset, Seaborn e Matplotlib para visualizações gráficas, imblearn para balanceamento com SMOTE e XGBoost para construção do modelo de classificação.
```
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import xgboost as xgb 

# Carregando o dataset
file_path = 'C:/Users/tulio/Downloads/dados_tratados_V2.csv'
data = pd.read_csv(file_path, delimiter=';')
```

O trecho de código a seguir é responsável por carregar algumas informações básicas do dataset

```
print(data.head())
print(data.info())
```

Neste trecho, são exploradas a distribuição dos estilos musicais e a relação entre a frequência de gêneros musicais e os efeitos percebidos.
```
style_columns = [col for col in data.columns if 'frequency' in col]
data[style_columns].hist(bins=15, figsize=(15, 10))
plt.show()

sns.countplot(x='Music effects', data=data)
plt.show()
```

O código remove os registros com valores nulos na variável alvo (Music effects) e realiza uma mapeação de categorias para valores numéricos:

> [!NOTE]
> Improve: 2 | No effect: 1 | Worsen: 0
```
data_clean = data.dropna(subset=['Music effects'])

data_clean['Music effects'] = data_clean['Music effects'].map({'Improve': 2, 'No effect': 1, 'Worsen': 0})
```

As colunas com informações sobre estilos musicais e fatores psicológicos foram selecionadas como features (X), e a variável Music effects foi definida como a variável alvo (y).
```
X = data_clean[style_columns + ['Anxiety', 'Depression', 'Insomnia', 'OCD', 'Age']]
y = data_clean['Music effects']
```

Para lidar com o desbalanceamento entre classes na variável alvo, utilizou-se o algoritmo SMOTE (Synthetic Minority Oversampling Technique)
```
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

Os dados foram divididos em conjuntos de treino e teste, preservando a distribuição proporcional das classes (stratify)
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Foi definida uma grade de hiperparâmetros para otimização do modelo XGBoost. Os parâmetros incluem:

- Número de árvores (n_estimators).
- Profundidade máxima das árvores (max_depth).
- Taxa de aprendizado (learning_rate).
- Amostras utilizadas em cada árvore (subsample).
- Proporção de amostras usadas por árvore na construção de árvores (colsample_bytree).
- Peso para lidar com desbalanceamento (scale_pos_weight).
```
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [1, 2, 3]  # para lidar com desbalanceamento
}
```

A otimização foi realizada utilizando o GridSearchCV, com 5 dobras de validação cruzada e a métrica F1 Macro. O melhor modelo foi selecionado com base nos parâmetros encontrados.
```
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
grid_search = GridSearchCV(
    xgb_model,
    param_grid,
    cv=5,
    scoring='f1_macro',
    verbose=1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
```

O modelo foi avaliado no conjunto de teste, e as métricas de qualidade foram calculadas, incluindo a matriz de confusão.
```
y_pred = best_model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(pd.crosstab(y_test, y_pred, rownames=['Real'], colnames=['Previsto']), annot=True, cmap='Blues', fmt='d')
plt.title("Matriz de Confusão")
plt.show()

print("Melhores parâmetros:", grid_search.best_params_)
```

A estrutura da árvore de decisão foi visualizada para melhor interpretar as decisões tomadas pelo modelo
```
plt.figure(figsize=(25, 12))
xgb.plot_tree(best_model, num_trees=0)  # A árvore de decisão número 0 do modelo XGBoost
plt.title('Visualização da Árvore de Decisão do Modelo XGBoost')
plt.show()
```

# CASO 2 - Árvore de Decisão

## Preparação dos dados

## Descrição dos modelos utilizados

## Importação da bibliotecas necessárias e carregamento do dataset

## Criação de modelos preditivos

### Random Forest

#### Conclusões

### Random Forest com Undersampling

#### Conclusões

### Random Forest com Oversampling

#### Conclusões

### XGBoost

#### Conclusões

### Random Forest com SMOTE (Synthetic Minority Over-sampling Technique)

#### Conclusões

### Conclusão sobre os modelos utilizados

## Análise dos efeitos da música por faixa etária

### Conclusões sobre a análise

## Correlação entre os estilos musicais com a melhoria na condição mental

### Conclusões sobre a análise

## Conclusão do Caso 2

### Conclusões técnicas

### Considerações finais





# Preparação dos dados

Nesta etapa, deverão ser descritas todas as técnicas utilizadas para pré-processamento/tratamento dos dados.

Algumas das etapas podem estar relacionadas à:

* Limpeza de Dados: trate valores ausentes: decida como lidar com dados faltantes, seja removendo linhas, preenchendo com médias, medianas ou usando métodos mais avançados; remova _outliers_: identifique e trate valores que se desviam significativamente da maioria dos dados.

* Transformação de Dados: normalize/padronize: torne os dados comparáveis, normalizando ou padronizando os valores para uma escala específica; codifique variáveis categóricas: converta variáveis categóricas em uma forma numérica, usando técnicas como _one-hot encoding_.

* _Feature Engineering_: crie novos atributos que possam ser mais informativos para o modelo; selecione características relevantes e descarte as menos importantes.

* Tratamento de dados desbalanceados: se as classes de interesse forem desbalanceadas, considere técnicas como _oversampling_, _undersampling_ ou o uso de algoritmos que lidam naturalmente com desbalanceamento.

* Separação de dados: divida os dados em conjuntos de treinamento, validação e teste para avaliar o desempenho do modelo de maneira adequada.
  
* Manuseio de Dados Temporais: se lidar com dados temporais, considere a ordenação adequada e técnicas específicas para esse tipo de dado.
  
* Redução de Dimensionalidade: aplique técnicas como PCA (Análise de Componentes Principais) se a dimensionalidade dos dados for muito alta.

* Validação Cruzada: utilize validação cruzada para avaliar o desempenho do modelo de forma mais robusta.

* Monitoramento Contínuo: atualize e adapte o pré-processamento conforme necessário ao longo do tempo, especialmente se os dados ou as condições do problema mudarem.

* Entre outras....

Avalie quais etapas são importantes para o contexto dos dados que você está trabalhando, pois a qualidade dos dados e a eficácia do pré-processamento desempenham um papel fundamental no sucesso de modelo(s) de aprendizado de máquina. É importante entender o contexto do problema e ajustar as etapas de preparação de dados de acordo com as necessidades específicas de cada projeto.

# Descrição dos modelos

Nesta seção, conhecendo os dados e de posse dos dados preparados, é hora de descrever os algoritmos de aprendizado de máquina selecionados para a construção dos modelos propostos. Inclua informações abrangentes sobre cada algoritmo implementado, aborde conceitos fundamentais, princípios de funcionamento, vantagens/limitações e justifique a escolha de cada um dos algoritmos. 

Explore aspectos específicos, como o ajuste dos parâmetros livres de cada algoritmo. Lembre-se de experimentar parâmetros diferentes e principalmente, de justificar as escolhas realizadas.

Como parte da comprovação de construção dos modelos, um vídeo de demonstração com todas as etapas de pré-processamento e de execução dos modelos deverá ser entregue. Este vídeo poderá ser do tipo _screencast_ e é imprescindível a narração contemplando a demonstração de todas as etapas realizadas.

# Avaliação dos modelos criados

## Métricas utilizadas

Nesta seção, as métricas utilizadas para avaliar os modelos desenvolvidos deverão ser apresentadas (p. ex.: acurácia, precisão, recall, F1-Score, MSE etc.). A escolha de cada métrica deverá ser justificada, pois esta escolha é essencial para avaliar de forma mais assertiva a qualidade do modelo construído. 

## Discussão dos resultados obtidos

Nesta seção, discuta os resultados obtidos pelos modelos construídos, no contexto prático em que os dados se inserem, promovendo uma compreensão abrangente e aprofundada da qualidade de cada um deles. Lembre-se de relacionar os resultados obtidos ao problema identificado, a questão de pesquisa levantada e estabelecendo relação com os objetivos previamente propostos. 

# Pipeline de pesquisa e análise de dados

Em pesquisa e experimentação em sistemas de informação, um pipeline de pesquisa e análise de dados refere-se a um conjunto organizado de processos e etapas que um profissional segue para realizar a coleta, preparação, análise e interpretação de dados durante a fase de pesquisa e desenvolvimento de modelos. Esse pipeline é essencial para extrair _insights_ significativos, entender a natureza dos dados e, construir modelos de aprendizado de máquina eficazes. 

## Observações importantes

Todas as tarefas realizadas nesta etapa deverão ser registradas em formato de texto junto com suas explicações de forma a apresentar  os códigos desenvolvidos e também, o código deverá ser incluído, na íntegra, na pasta "src".

Além disso, deverá ser entregue um vídeo onde deverão ser descritas todas as etapas realizadas. O vídeo, que não tem limite de tempo, deverá ser apresentado por **todos os integrantes da equipe**, de forma que, cada integrante tenha oportunidade de apresentar o que desenvolveu e as  percepções obtidas.

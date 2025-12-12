import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

print("--- INICIANDO TREINAMENTO COM SCIKIT-LEARN ---")

# 1. Carregar os dados
df = pd.read_csv('Text_Emotion.csv')
# Ele converte para minúsculas e remove pontuação por padrão.

# 2. Preparar os dados e aplicar TF-IDF
# TfidfVectorizer transforma todo o texto em uma matriz de features numéricas
# Remove stopwords em inglês
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000, # Limita o vocabulário às 5000 palavras mais importantes
    ngram_range=(1, 2) # Usa unigramas e bigramas
)

# Cria a matriz de features (X) e o vetor de alvos (y)
X = vectorizer.fit_transform(df['text'])
y = df['emotion']

# 3. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Treinar e avaliar múltiplos modelos
models = {
    "Naive Bayes": MultinomialNB(),
    "Regressão Logística": LogisticRegression(solver='saga', max_iter=1000), 
    "SVM (LinearSVC)": LinearSVC(max_iter=2000)
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    print(f"\n>>> Treinando modelo: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do {name}: {accuracy * 100:.2f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# 5. Salvar o melhor modelo e o vetorizador
print(f"\nMelhor modelo foi o {type(best_model).__name__} com acurácia de {best_accuracy * 100:.2f}%")
print("Salvando o melhor modelo e o vetorizador...")

with open('sklearn_classifier.pickle', 'wb') as f:
    pickle.dump(best_model, f)
    
with open('tfidf_vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Modelo e vetorizador salvos com sucesso!")
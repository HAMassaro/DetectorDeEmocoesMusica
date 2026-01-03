import sys
import pickle
import numpy as np
from collections import Counter


# 1. Verifica se o argumento do arquivo foi fornecido
if len(sys.argv) < 2:
    print("ERRO: Argumento faltando.")
    print("Uso: python classificador_por_argumento.py <caminho_para_o_arquivo.txt>")
    sys.exit() # Encerra o script se nenhum arquivo for passado

# 3. Pega o nome do arquivo do primeiro argumento
NOME_ARQUIVO = sys.argv[1]

# 4. Verifica se o arquivo existe
try:
    with open('sklearn_classifier.pickle', 'rb') as f:
        classifier = pickle.load(f)
    with open('tfidf_vectorizer.pickle', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Modelo Scikit-learn carregado.")
    CLASS_NAMES = classifier.classes_
except FileNotFoundError:
    print("ERRO: Arquivos de modelo não encontrados. Execute o script de treinamento primeiro.")
    exit()

# 5. Funções para classificar frases e obter scores
def classificar_frase_sklearn(frase):
    vetor_frase = vectorizer.transform([frase])
    return classifier.predict(vetor_frase)[0]

def obter_scores_frase(frase):
    vetor_frase = vectorizer.transform([frase])
    return classifier.predict_proba(vetor_frase)[0]

lista_de_emocoes = []
lista_de_scores = []

print(f"\n>>> Analisando a letra do arquivo: '{NOME_ARQUIVO}'\n")

# 6. Lê o arquivo de letra da música
try:
    with open(NOME_ARQUIVO, 'r', encoding='utf-8') as f:
        versos_da_musica = f.readlines()

    if not any(verso.strip() for verso in versos_da_musica):
        print("O arquivo de letra está vazio.")
        exit()

    print("--- 1. Jornada Emocional da Música ---")
    for verso in versos_da_musica:
        verso = verso.strip()
        if verso:
            emocao_verso = classificar_frase_sklearn(verso)
            lista_de_emocoes.append(emocao_verso)
            scores_verso = obter_scores_frase(verso)
            lista_de_scores.append(scores_verso)
            print(f"'{verso}': -> {emocao_verso.upper()}")
    
    print("\n" + "="*40)
    print("      RESULTADO FINAL DA ANÁLISE")
    print("="*40)


    contagem = Counter(lista_de_emocoes)
    emocao_geral = contagem.most_common(1)[0][0]
    print("\n--- 2. Análise por Voto da Maioria ---")
    print(f"A emoção dominante da música é: '{emocao_geral}'")
    print(f"Contagem detalhada das emoções: {dict(contagem)}")

    score_medio = np.mean(lista_de_scores, axis=0)
    print("\n--- 3. Análise por Score de Emoção (Média das Probabilidades) ---")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"Score de '{class_name}': {score_medio[i] * 100:.2f}%")

except FileNotFoundError:
    print(f"\nERRO: O arquivo '{NOME_ARQUIVO}' não foi encontrado.")
    print("Verifique se o caminho e o nome do arquivo estão corretos.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")
import cv2 as cv
import numpy as np
import os

class TreinarModelo:

    def __init__(self):

        self.raiz_projeto = os.getcwd()

    def carregar_imagens(self):
        
        imagens = []
        rotulos = []

        diretorio_imagens = os.path.join(self.raiz_projeto, 'imagens')

        for nome_pessoa in os.listdir(diretorio_imagens):
            pasta_pessoa = os.path.join(diretorio_imagens, nome_pessoa)

            if not os.path.isdir(pasta_pessoa):
                continue

            # Usando o nome da pessoa como rótulo
            rotulo = nome_pessoa

            for nome_arquivo in os.listdir(pasta_pessoa):
                caminho_imagem = os.path.join(pasta_pessoa, nome_arquivo)

                imagem = cv.imread(caminho_imagem, cv.IMREAD_GRAYSCALE)

                if imagem is None:
                    continue

                imagens.append(imagem)
                rotulos.append(rotulo)

        return imagens, np.array(rotulos)

    def treinar_modelo(self):
        
        imagens, rotulos = self.carregar_imagens()

        # Criação do modelo LBPH (Local Binary Patterns Histograms)
        modelo = cv.face.LBPHFaceRecognizer_create()

        # Convertendo os rótulos para números
        unique_labels = np.unique(rotulos)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        rotulos_numericos = np.array([label_map[label] for label in rotulos])

        # Treinamento do modelo
        modelo.train(imagens, rotulos_numericos)

        # Salvando o modelo treinado e o mapa de rótulos
        modelo.save('modelo_treinado.xml')
        np.save('label_map.npy', label_map)

        print("Modelo treinado!")

# Exemplo de uso da classe TreinaModelo
if __name__ == "__main__":
    trainer = TreinarModelo()
    trainer.treinar_modelo()
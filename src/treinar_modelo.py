import cv2 as cv
import numpy as np
import os

class TreinaModelo:
    def __init__(self, diretorio_base):
        self.diretorio_base = diretorio_base

    def carregar_imagens(self):
        imagens = []
        rotulos = []
        diretorio_imagens = os.path.join(self.diretorio_base, 'imagens')

        for nome_pessoa in os.listdir(diretorio_imagens):
            pasta_pessoa = os.path.join(diretorio_imagens, nome_pessoa)
            if not os.path.isdir(pasta_pessoa):
                continue
            
            # Extraindo o rótulo da pessoa do nome da pasta, assumindo que a pasta segue o padrão `nome_X`
            rotulo = int(nome_pessoa.split('_')[1])
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

        if len(imagens) == 0:
            raise Exception("Nenhuma imagem foi carregada. Verifique a estrutura do diretório e os arquivos de imagem.")

        # Criação do modelo LBPH (Local Binary Patterns Histograms)
        modelo_lbph = cv.face.LBPHFaceRecognizer_create()

        # Treinamento do modelo
        modelo_lbph.train(imagens, rotulos)

        # Salvando o modelo treinado
        modelo_lbph.save('modelo_lbph.xml')

        print("Modelo treinado e salvo com sucesso!")

# Exemplo de uso da classe TreinaModelo
if __name__ == "__main__":
    diretorio_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'visao_computacional'))
    trainer = TreinaModelo(diretorio_base)
    trainer.treinar_modelo()

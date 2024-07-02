import cv2 as cv
import numpy as np

class ReconhecimentoFacial:

    def __init__(self):

        self.video = cv.VideoCapture(0)  # objeto que inicializa a camera principal (0)

        self.classificador_face = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')  # modelo pre treinado

        # Carregar o modelo de reconhecimento facial previamente treinado
        self.modelo_reconhecimento = cv.face.LBPHFaceRecognizer_create()
        self.modelo_reconhecimento.read('modelo_treinado.xml')
        self.label_map = np.load('label_map.npy', allow_pickle=True).item()
        self.label_map_inverso = {v: k for k, v in self.label_map.items()}

    def detectar_face(self):

        while True:

            ret, frame = self.video.read()  # inicia a leitura de tela

            if not ret:  # verifica se a camera abriu corretamente
                break

            gray = self.processar_imagem(frame)

            faces = self.classificador_face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)  # faz o reconhecimento do rosto

            for (x, y, w, h) in faces:  # desenha um retangulo em volta do rosto

                cv.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Reconhecimento facial
                rosto = gray[y:y + h, x:x + w]
                nome_pessoa, confianca = self.reconhecer_face(rosto)

                # Exibir nome da pessoa reconhecida
                cv.putText(gray, f'{nome_pessoa} ({confianca:.2f})', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv.imshow("janela", gray)  # exibe a janela da camera

            if cv.waitKey(1) == ord('q'):  # encerra a execução
                break

        self.video.release()  # encerra a conexão com a camera

        cv.destroyAllWindows()  # destroi todas as janelas existentes

    def processar_imagem(self, frame):
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # converte os frames em cinza

        return gray

    def reconhecer_face(self, rosto):
        rotulo, confianca = self.modelo_reconhecimento.predict(rosto)
        nome_pessoa = self.label_map_inverso.get(rotulo, "Desconhecido")
        return nome_pessoa, confianca

# Exemplo de uso da classe ReconhecimentoFacial
if __name__ == "__main__":
    reconhecimento = ReconhecimentoFacial()
    reconhecimento.detectar_face()

import cv2 as cv

class ReconhecimentoFacial:

    def __init__ (self):
        self.video = cv.VideoCapture (0) # objeto que inicializa a camera principal (0)
        self.classificador_face = cv.CascadeClassifier (cv.data.haarcascades + 'haarcascade_frontalface_default.xml') # modelo pre treinado

    def detectar_face (self):

        while True:
           
           ret, frame = self.video.read () # inicia a leitura de tela

           if not ret: # verifica se a camera abriu corretamente
                break
           
           gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # converte os frames em cinza

           faces = self.classificador_face.detectMultiScale (gray, scaleFactor=1.3, minNeighbors=5) # faz o reconhecimento do rosto

           for (x, y, w, h) in faces: # desenha um retangulo em volta do rosto
                cv.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
           
           cv.imshow ("janela", gray) # exibe a janela da camera 

           if cv.waitKey(1) == ord('q'): # encerra a execução
                break
           
        self.video.release () # encerra a conexão com a camera
        cv.destroyAllWindows () # destroi todas as janelas existentes
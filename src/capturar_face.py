import cv2 as cv

class capturar_face:

    def __init__(self):
        
        self.video = cv.VideoCapture (0) # objeto que inicializa a camera principal (0)
        self.classificador_face = cv.CascadeClassifier (cv.data.haarcascades + 'haarcascade_frontalface_default.xml') # modelo pre treinado

    def iniciar_camera (self):
        
        while (True):

            ret, frame = self.video.read () # Inicia a leitura pela camera

            if (not ret): # verifica se o frame pode ser capturado
                break

            gray = cv.cvtColor (self.detectar_face(frame), cv.COLOR_BGR2GRAY) # processamento de imagem, que transforma em escala cinza

            cv.imshow ('Janela', gray) # exibe a camera

            if cv.waitKey (1) & 0xFF == ord('q'): # condicional para encerrar a execução
                break

        # Liberação de recursos

        self.video.release () # encerra a conexão com a camera
        cv.destroyAllWindows () # fecha todas as janelas aberta pelo open-cv

    
    def detectar_face (self, imagem):

        gray = cv.cvtColor (imagem, cv.COLOR_BGR2GRAY) # processamento de imagem, que transforma em escala cinza

        faces = self.classificador_face.detectMultiScale (gray, scaleFactor=1.3, minNeighbors=5) # detecta o rosto

        for (x, y, w, h) in faces: # desenha um retangulo em volta das faces detectadas
            cv.rectangle(imagem, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return imagem
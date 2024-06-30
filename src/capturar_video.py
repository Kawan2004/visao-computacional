import cv2 as cv

class capturar_video:

    def __init__(self):
        
        self.video = cv.VideoCapture (0) # objeto que inicializa a camera principal (0)

    def iniciar_camera (self):
        
        while (True):

            ret, frame = self.video.read () # Inicia a leitura pela camera

            if (not ret): # verifica se o frame pode ser capturado
                break

            gray = cv.cvtColor (frame, cv.COLOR_BGR2GRAY) # processamento de imagem, que transforma em escala cinza

            cv.imshow ('frame', gray) # exibe a camera

            if cv.waitKey (1) & 0xFF == ord('q'): # condicional para encerrar a execução
                break

        # Liberação de recursos

        self.video.release () # encerra a conexão com a camera
        cv.destroyAllWindows () # fecha todas as janelas aberta pelo open-cv
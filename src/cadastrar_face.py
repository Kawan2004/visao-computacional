import cv2 as cv
import os

class CadastrarFace:

    def __init__(self, nome):
        
        self.video = cv.VideoCapture (0) # objeto que inicializa a camera principal (0)
        self.contador_imagem = 0
        self.nome = nome

    def capturar_imagens (self):
        
        while (True):

            ret, frame = self.video.read () # Inicia a leitura pela camera

            if (not ret): # verifica se o frame pode ser capturado
                break

            cv.imshow ('Janela', self.processar_imagem (frame)) # exibe a camera

            if cv.waitKey (1) & 0xFF == ord('c'): # condicional para capturar rosto



                self.salvar_imagem (frame)
                
            if cv.waitKey (1) & 0xFF == ord('q'): # condicional para encerrar a execução
                break

        self.video.release () # encerra a conexão com a camera
        cv.destroyAllWindows () # fecha todas as janelas aberta pelo open-cv

    def processar_imagem (self, frame):

        gray = cv.cvtColor (frame, cv.COLOR_BGR2GRAY) # processamento de imagem, que transforma em escala cinza

        return gray

    def salvar_imagem (self, frame):

        os.makedirs("imagens", exist_ok = True) # criando diretorio para guardar imagens

        os.makedirs(f"imagens/{self.nome}", exist_ok = True) # criando diretorio para guardar imagens

        nome_imagem = f"img-{self.contador_imagem}.jpg"

        path = os.path.join(f"imagens/{self.nome}", nome_imagem)

        cv.imwrite(path, self.processar_imagem (frame)) # metodo para guardar imagens

        print(f"imagem capturada com sucesso!")

        self.contador_imagem += 1  # Incrementar o contador para o próximo arquivo'''

     
        
        

        

        

        

    


    
    
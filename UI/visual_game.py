import pygame
import numpy as np
import sys

# Configurazioni Layout
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600
SPLIT_X = 600  # Limite tra gioco (sx) e testo (dx)
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (200, 200, 200)
ACCENT_COLOR = (0, 255, 127) # Verde brillante per titoli
HEADER_COLOR = (255, 215, 0) # Oro per i titoli degli episodi
SEPARATOR_COLOR = (80, 80, 80)
FONT_SIZE = 18 # Leggermente più piccolo per farci stare più testo

class LayoutManager:
    def __init__(self, env_id):
        pygame.init()
        pygame.display.set_caption(f"Reviewer Training - {env_id}")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.font = pygame.font.SysFont("Consolas", FONT_SIZE)
        self.title_font = pygame.font.SysFont("Consolas", FONT_SIZE + 2, bold=True)
        self.clock = pygame.time.Clock()
        
        # BUFFER: Lista di dizionari [{'id': ep_num, 'text': [lines]}]
        # Mantiene massimo 2 elementi (episodio precedente, episodio corrente)
        self.episode_buffer = [] 

    def update_text(self, text_lines, episode_num=1):
        """
        Aggiorna il testo. Gestisce il buffer degli episodi.
        episode_num: serve per capire se è un nuovo episodio o l'aggiornamento di quello corrente.
        """
        # Normalizza input in lista
        if isinstance(text_lines, str):
            lines = [text_lines]
        elif isinstance(text_lines, list):
            lines = text_lines
        else:
            lines = [str(text_lines)]

        # LOGICA BUFFER
        # Caso 1: Buffer vuoto o Nuovo Episodio (ID diverso dall'ultimo nel buffer)
        if not self.episode_buffer or self.episode_buffer[-1]['id'] != episode_num:
            self.episode_buffer.append({'id': episode_num, 'text': lines})
            
            # Se abbiamo più di 2 episodi, rimuoviamo il più vecchio (il primo della lista)
            if len(self.episode_buffer) > 2:
                self.episode_buffer.pop(0)
        
        # Caso 2: Aggiornamento dello stesso episodio
        else:
            self.episode_buffer[-1]['text'] = lines

    def render(self, env_img):
        """
        Disegna il frame corrente: Gioco a SX, Buffer Testuale a DX.
        """
        # 1. Gestione chiusura finestra
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(BG_COLOR)

        # 2. LATO SINISTRO: GIOCO
        if env_img is not None:
            # MiniGrid restituisce (W, H, 3), PyGame vuole (W, H)
            surf = pygame.surfarray.make_surface(np.transpose(env_img, (1, 0, 2)))
            surf = pygame.transform.scale(surf, (SPLIT_X, SCREEN_HEIGHT))
            self.screen.blit(surf, (0, 0))

        # Linea divisoria verticale
        pygame.draw.line(self.screen, (100, 100, 100), (SPLIT_X, 0), (SPLIT_X, SCREEN_HEIGHT), 3)

        # 3. LATO DESTRO: RENDER DEL BUFFER
        x_offset = SPLIT_X + 20
        y_offset = 20
        
        # Cicla attraverso gli episodi nel buffer (massimo 2)
        for entry in self.episode_buffer:
            ep_id = entry['id']
            lines = entry['text']

            # --- Disegna Intestazione Episodio ---
            header_surf = self.title_font.render(f"=== EPISODIO {ep_id} ===", True, HEADER_COLOR)
            self.screen.blit(header_surf, (x_offset, y_offset))
            y_offset += 30

            # --- Disegna il testo dell'episodio ---
            for line in lines:
                words = str(line).split(' ')
                current_line = ""
                for word in words:
                    test_line = current_line + word + " "
                    # Word wrapping
                    if self.font.size(test_line)[0] < (SCREEN_WIDTH - SPLIT_X - 40):
                        current_line = test_line
                    else:
                        line_surf = self.font.render(current_line, True, TEXT_COLOR)
                        self.screen.blit(line_surf, (x_offset, y_offset))
                        y_offset += 20 # Interlinea
                        current_line = word + " "
                
                if current_line:
                    line_surf = self.font.render(current_line, True, TEXT_COLOR)
                    self.screen.blit(line_surf, (x_offset, y_offset))
                    y_offset += 20
            
            # Spazio extra e linea separatrice tra episodi
            y_offset += 10
            pygame.draw.line(self.screen, SEPARATOR_COLOR, (SPLIT_X + 10, y_offset), (SCREEN_WIDTH - 10, y_offset), 1)
            y_offset += 20 # Spazio per il prossimo blocco

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
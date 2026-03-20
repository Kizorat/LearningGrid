import pygame
import numpy as np

# Layout configuration
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600
SPLIT_X = 600  # Split between game area (left) and text area (right)
DIVIDER_WIDTH = 3
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (200, 200, 200)
ACCENT_COLOR = (0, 255, 127) # Bright green for titles
HEADER_COLOR = (255, 215, 0) # Gold for episode titles
SEPARATOR_COLOR = (80, 80, 80)
FONT_SIZE = 18 # Slightly smaller to fit more text

class LayoutManager:
    def __init__(self, env_id):
        pygame.init()
        pygame.display.set_caption(f"Reviewer Training - {env_id}")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.font = pygame.font.SysFont("Consolas", FONT_SIZE)
        self.title_font = pygame.font.SysFont("Consolas", FONT_SIZE + 2, bold=True)
        self.clock = pygame.time.Clock()
        
        # Buffer format: list of dicts [{'id': ep_num, 'text': [lines]}]
        # Keep at most 2 elements (previous episode and current episode)
        self.episode_buffer = []
        self.game_rect = pygame.Rect(0, 0, SPLIT_X - DIVIDER_WIDTH, SCREEN_HEIGHT)
        self.running = True
        self.last_frame_surface = None

    def set_env(self, env_id):
        pygame.display.set_caption(f"Reviewer Training - {env_id}")

    def update_text(self, text_lines, episode_num=1):
        # Normalize input to a list
        if isinstance(text_lines, str):
            lines = [text_lines]
        elif isinstance(text_lines, list):
            lines = text_lines
        else:
            lines = [str(text_lines)]

        # Buffer logic
        # Empty buffer or new episode (ID different from last entry)
        if not self.episode_buffer or self.episode_buffer[-1]['id'] != episode_num:
            # New episode, first keep only one previous episode
            if len(self.episode_buffer) >= 2:
                # Remove oldest entry and keep only the most recent one
                self.episode_buffer = [self.episode_buffer[-1]]
            
            # Add the new episode
            self.episode_buffer.append({'id': episode_num, 'text': lines})
        
        # Update current episode
        else:
            self.episode_buffer[-1]['text'] = lines

    def render(self, env_img):
        # Handle window close events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False

        if not self.running:
            return False

        # Convert new frame if available; otherwise keep the last valid one.
        if env_img is not None:
            frame = np.asarray(env_img)
            if frame.ndim == 3 and frame.shape[2] >= 3:
                frame = frame[:, :, :3]
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                frame = np.ascontiguousarray(np.transpose(frame, (1, 0, 2)))
                self.last_frame_surface = pygame.surfarray.make_surface(frame)

        self.screen.fill(BG_COLOR)

        # Left side: game view (single source of truth for MiniGrid frame rendering)
        pygame.draw.rect(self.screen, (0, 0, 0), self.game_rect)
        if self.last_frame_surface is not None:
            src_w = self.last_frame_surface.get_width()
            src_h = self.last_frame_surface.get_height()
            if src_w > 0 and src_h > 0:
                scale = min(self.game_rect.width / src_w, self.game_rect.height / src_h)
                target_w = max(1, int(src_w * scale))
                target_h = max(1, int(src_h * scale))
                scaled = pygame.transform.smoothscale(self.last_frame_surface, (target_w, target_h))

                x = self.game_rect.x + (self.game_rect.width - target_w) // 2
                y = self.game_rect.y + (self.game_rect.height - target_h) // 2
                self.screen.set_clip(self.game_rect)
                self.screen.blit(scaled, (x, y))
                self.screen.set_clip(None)

        # Vertical divider line
        pygame.draw.line(
            self.screen,
            (100, 100, 100),
            (SPLIT_X - DIVIDER_WIDTH, 0),
            (SPLIT_X - DIVIDER_WIDTH, SCREEN_HEIGHT),
            DIVIDER_WIDTH,
        )

        # Right side: render text buffer
        x_offset = SPLIT_X + 20
        y_offset = 20
        
        # Iterate through buffered episodes (max 2)
        for entry in self.episode_buffer:
            ep_id = entry['id']
            lines = entry['text']

            # Draw episode header
            header_surf = self.title_font.render(f"=== EPISODE {ep_id} ===", True, HEADER_COLOR)
            self.screen.blit(header_surf, (x_offset, y_offset))
            y_offset += 30

            # Draw episode text
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
                        y_offset += 20 # Line spacing
                        current_line = word + " "
                
                if current_line:
                    line_surf = self.font.render(current_line, True, TEXT_COLOR)
                    self.screen.blit(line_surf, (x_offset, y_offset))
                    y_offset += 20
            
            # Extra spacing and separator line between episodes
            y_offset += 10
            pygame.draw.line(self.screen, SEPARATOR_COLOR, (SPLIT_X + 10, y_offset), (SCREEN_WIDTH - 10, y_offset), 1)
            y_offset += 20 # Space before next block

        pygame.display.flip()
        self.clock.tick(30)
        return True

    def close(self):
        self.running = False
        pygame.quit()
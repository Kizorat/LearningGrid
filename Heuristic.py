import gymnasium as gym
import minigrid
import time
import keyboard
import sys

# Mappa delle scelte e degli id registrati
env_options = {
    "1": {
        "label": "Crossing",
        "ids": [
            "MiniGrid-LavaCrossingS9N1-v0",
            "MiniGrid-LavaCrossingS9N3-v0",
            "MiniGrid-LavaCrossingS11N5-v0",
        ],
    },
    "2": {
        "label": "DoorKey",
        "ids": [
            "MiniGrid-DoorKey-5x5-v0",
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-DoorKey-16x16-v0",
        ],
    },
    "3": {
        "label": "Empty",
        "ids": [
            "MiniGrid-Empty-5x5-v0",
            "MiniGrid-Empty-8x8-v0",
            "MiniGrid-Empty-16x16-v0",
        ],
    },
}

# Scelta dell'ambiente
while True:
    choice = input("Scegli l'ambiente (1=Crossing, 2=DoorKey, 3=Empty): ").strip()
    if choice in env_options:
        break
    print("Scelta non valida, inserisci 1, 2 o 3.")

# Scelta della dimensione
while True:
    size = input("Scegli la dimensione della mappa (1=piccolo, 2=medio, 3=grande): ").strip()
    if size in ("1", "2", "3"):
        break
    print("Scelta non valida, inserisci 1, 2 o 3.")

env_id = env_options[choice]["ids"][int(size) - 1]
print(f"Ambiente selezionato: {env_options[choice]['label']} -> {env_id}")

# Crea l'ambiente con modalità render “human”
try:
    env = gym.make(env_id, render_mode="human")
except Exception as e:
    print(f"Errore creando l'ambiente {env_id}: {e}")
    sys.exit(1)

# Reset iniziale
obs, info = env.reset(seed=42)

def movement_command():
    if keyboard.is_pressed('esc'):
        print("Uscita.")
        exit(0)

    if keyboard.is_pressed('w'):
        action = 2
    elif keyboard.is_pressed('a'):
        action = 0
    elif keyboard.is_pressed('d'):
        action = 1
    elif keyboard.is_pressed('q'):
        action = 3
    elif keyboard.is_pressed('e'):
        action = 4
    elif keyboard.is_pressed('f'):
        action = 5
    elif keyboard.is_pressed('x'):
        action = 6
    else:
        time.sleep(0.001)
        action = 6 
    return action

print("Controllo manuale: w=forward, a=left, d=right, q=pickup, e=drop, f=toggle, x=done, esc=esci")

# Loop interattivo: leggere comandi da tastiera via terminale
while True:
    env.render()

    action = movement_command()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episodio terminato, reset.")
        obs, info = env.reset()
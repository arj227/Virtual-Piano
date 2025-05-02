import os
from playsound import playsound
from multiprocessing import Process
import keyboard

cwd   = os.path.dirname(os.path.abspath(__file__))
projectRoot = os.path.dirname(cwd)
pianoSoundsDir = os.path.join (projectRoot, 'data', 'piano_sounds')

KEY_MAP = {
    '1': 'A',
    '2': 'B',
    '3': 'C',
    '4': 'D',
    '5': 'E',
    '6': 'F',
    '7': 'G',
}

def play_note(note: str):
    path = os.path.join(pianoSoundsDir, f"{note}.wav")
    if not os.path.isfile(path):
        print(f"Sample not found: {path}")
        return
    
    p = Process(target=playsound, args=(path,), daemon=True)
    p.start()

def onKey(event):
    if event.event_type == 'down' and event.name in KEY_MAP:
        note = KEY_MAP[event.name]
        print(f"Key {event.name!r} → playing {note}")
        play_note(note)


if __name__ == '__main__':

    keyboard.hook(onKey)
    print("Press 1–7 to play notes; ESC to quit.")
    keyboard.wait('esc')



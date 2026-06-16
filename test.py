import os

filepaths = []

for actor in os.listdir("."):
    if actor.startswith("Actor_"):
        actor_path = os.path.join(".", actor)
        
        for f in os.listdir(actor_path):
            if f.endswith(".wav"):
                filepaths.append(os.path.join(actor_path, f))

print(len(filepaths))
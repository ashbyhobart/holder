import pickle

objects = []
with (open("results/ray_workers_8_bsize_3_actorfr_1.0.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

print(objects)
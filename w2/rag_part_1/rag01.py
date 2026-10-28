import numpy as np
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-50")

model["banana"]
v_banana = model["banana"]

v_grape = model["grape"]
v_aeroplane = model["aeroplane"]

sim_banana_all = model.cosine_similarities(v_banana, [v_banana, v_grape, v_aeroplane])
print("Cosine similarities of 'banana' with itself, 'grape', and 'aeroplane':", sim_banana_all)
v_king = model["king"]
v_queen = model["queen"]
v_man = model["man"]

new_word = model.most_similar(v_man + v_queen - v_king, topn=1)

v_paris = model["paris"]
v_france = model["france"]
v_india = model["india"]
new_word_2 = model.most_similar(v_paris - v_france + v_india, topn=1)

from movie_combine import combineMovieFiles
import os

gifs_dir = "/home/declan/Documents/code/rwg-benchmark/output/promo_gifs"


file_list = [
    os.path.join(gifs_dir, "Cartpole_0HL_overlaid.mp4"),
    os.path.join(gifs_dir, "mountaincar_0HL_overlaid.mp4"),
    os.path.join(gifs_dir, "Acrobot_2HL_4HU_overlaid.mp4"),
    os.path.join(gifs_dir, "pendulum_1HL_4HU_overlaid.mp4"),
]

combineMovieFiles(
    path=gifs_dir,
    file_list=file_list,
    grid_size="2x2",
    make_gif=True,
    margin_size=0,
    bg_color=(255, 255, 255),
)

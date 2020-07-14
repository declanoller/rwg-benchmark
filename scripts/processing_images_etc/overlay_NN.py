
import subprocess
import os
#from PIL import Image, ImageDraw, ImageFile


def overlay_img(movie_fname, img_fname):


    fname_output = movie_fname.replace('.mp4', '_overlaid.mp4')

    x_coord = 0
    y_coord = 0

    overlay_cmd = 'ffmpeg -i {} -i {} -filter_complex "[0:v][1:v] overlay={}:{}:enable=\'between(t,0,20)\'" -pix_fmt yuv420p -c:a copy {}'.format(movie_fname, img_fname, x_coord, y_coord, fname_output)


    os.system(overlay_cmd)





gifs_dir = '/home/declan/Documents/code/rwg-benchmark/output/promo_gifs'

movie_fname = os.path.join(gifs_dir, 'MountainCarContinuous_2HL_4HU.mp4')
img_fname = os.path.join(gifs_dir, 'MC_cont_NN.png')

overlay_img(movie_fname, img_fname)

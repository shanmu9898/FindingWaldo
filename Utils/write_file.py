import os

'''
Ideally file_name should be the {name of the character}.txt
i.e. waldo.txt, wenda.txt or wizard.txt
Be careful to append to an existing file
'''
def write_file(file_path, img_name, score, xmin, ymin, xmax, ymax):
    # if file already exist, append; otherwise, create new
    if os.path.exists(file_path):
        append_write = 'a'
    else:
        append_write = 'w'

    with open(file_path, append_write) as f:
        f.write(f'{img_name} {score} {xmin} {ymin} {xmax} {ymax}\n')

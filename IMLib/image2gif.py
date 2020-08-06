import imageio

filename = '../test_dir/05_hi_inter.jpg'
with imageio.get_writer('../test_dir/movie.gif', mode='I') as writer:
    image = imageio.imread(filename)
    w = image.shape[1]
    im_list = []
    for i in range(w//1024):
        image0 = image[:, i*1024:(i+1)*1024,:]
        im_list.append(image0)
    im_list_ = im_list[1:] + list(reversed(im_list[1:]))
    for i in range(len(im_list_)):
        writer.append_data(im_list_[i])

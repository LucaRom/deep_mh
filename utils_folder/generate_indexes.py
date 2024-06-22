import rasterio
import numpy as np
from tqdm import tqdm
from rasterio import windows
from itertools import product

# The offset range are reajusted with the step size to avoid extra tiles beng created when reaching the end of the image
# but the start position of the tile is still inside.
# TODO see if there is any case that strict_shape would not handle all boundless cases
#def iter_windows(src_ds, stepsize, width, height, strict_shape=True, boundless=False):
def iter_windows(src_ds, stepsize, width, height, strict_shape=True, boundless=True):
    # offsets creates tuples for col_off, row_off
    #offsets = product(range(0, src_ds.meta['width']-stepsize, stepsize), range(0, src_ds.meta['height']-stepsize, stepsize))
    offsets = product(range(0, src_ds.meta['height']-128, stepsize), range(0, src_ds.meta['width']-128, stepsize))
    big_window = windows.Window(col_off=0, row_off=0, width=src_ds.meta['width'], height=src_ds.meta['height'])


    # Creates windows from offsets as start pixels and uses specified window size
    # You can switch col_off and row_off depending of the wanted sliding window direction
    #for col_off, row_off in offsets:
    for row_off, col_off in offsets:
        #print(col_off, row_off)
        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)
        transform = windows.transform(window, src_ds.transform)

        # if col_off > src_ds.meta['width']+256:
        #     print()

        if boundless:
            window = window
        else:
            window = window.intersection(big_window)

        # Strict shape limits output windows with only width x height shape
        if strict_shape:
            #if window.width < width or window.height < height:
            if row_off + 256 > src_ds.meta['width'] or col_off + 256 > src_ds.meta['height']:
                #print(col_off, row_off)
                pass
            else:
                yield window
        else:
            yield window
        #         pass
        #     else:
        #         yield window, transform
        # else:
        #     yield window, transform

def generate_indexes(img_path, wd_size, step_size, trim_background=True, save_file_version='v1000000'):
    with rasterio.open(img_path) as ds:
        windows_num = len(list(iter_windows(ds, step_size, wd_size, wd_size, strict_shape=False)))
        #indices_ori = list(range(windows_num))
        test_indices = []
        train_indices = []
        full_nh_tiles_idx = []
        bad_conf_list = []
        removed = []
        for idx, a_window in tqdm(enumerate(iter_windows(ds, step_size, wd_size, wd_size, strict_shape=False)), total=windows_num, desc='Windows'):
                mask = ds.read(1, window=a_window)
                zones = ds.read(2, window=a_window)
                #bad_conf = ds.read(3, window=a_window)

                nh_pixels = np.count_nonzero(mask == 7)
                test_pixels = np.count_nonzero(zones == 1)
                #bad_conf_pixels = np.count_nonzero(bad_conf != 7)

                # DEBUG
                #np.unique(bad_conf, return_counts=True)

                tile_pixels_num = wd_size * wd_size
                if a_window.col_off + 256 > ds.meta['width'] or a_window.row_off + 256 > ds.meta['height']:
                    removed.append(idx)
                else:
                    if test_pixels == tile_pixels_num:
                        test_indices.append(idx)
                    # elif bad_conf_pixels > 10:
                    #     bad_conf_list.append(idx)
                    #     #print(idx, bad_conf_pixels)
                    elif trim_background and nh_pixels == tile_pixels_num:
                        full_nh_tiles_idx.append(idx)
                    else:
                        train_indices.append(idx)

        #print('Original indices len :', len(indices_ori))
        print('Train val idx len:', len(train_indices))
        print('Test idx len :', len(test_indices))
        print('Removed idx len :', len(removed))
        print('Bad_conf len :', len(bad_conf_list))
        print('Total kept idx :', len(train_indices) + len(test_indices))
        print('Total number of idx :', len(train_indices) + len(test_indices) + len(removed) + len(bad_conf_list))
        print('Full NH idx len :', len(full_nh_tiles_idx))

        # if trim_background:
        #     trainval_idx_path = 'results/estrie_trainval_idx_list_trimmed'
        #     test_idx_path = 'results/estrie_test_idx_list_trimmed'
        # else :
        trainval_idx_path = 'results/estrie_trainval_idx_' + save_file_version
        test_idx_path = 'results/estrie_test_idx_' + save_file_version

        np.save(trainval_idx_path, train_indices)
        np.save(test_idx_path, test_indices)


if __name__ == '__main__':
     
    img_path = 'results/stack_mask3223_testzone_v03.tif'

    save_file_version = 'v15_trimmed_v2'

    generate_indexes(img_path, wd_size=256, step_size=128, trim_background=False, save_file_version=save_file_version)

    # print('test idx lenght :', len(np.load('results/estrie_test_idx_v16.npy')))
    # print('train idx lenght :', len(np.load('results/estrie_trainval_idx_v16.npy')))
from itertools import product
from rasterio import windows

def iter_windows_rasterio(src_ds, width, height, boundless=False):
    offsets = product(range(0, src_ds.meta['width'], width), range(0, src_ds.meta['height'], height))
    big_window = windows.Window(col_off=0, row_off=0, width=src_ds.meta['width'], height=src_ds.meta['height'])
    for col_off, row_off in offsets:

        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)

        if boundless:
            yield window
        else:
            yield window.intersection(big_window)

def iter_windows_rasterio_shape(src_ds, width, height, boundless=False):
    offsets = product(range(0, src_ds.shape[0], width), range(0, src_ds.shape[1], height))
    big_window = windows.Window(col_off=0, row_off=0, width=src_ds.shape[0], height=src_ds.shape[1])
    for col_off, row_off in offsets:

        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)

        if boundless:
            yield window
        else:
            yield window.intersection(big_window)

# INCOMPLETE
def create_rasterio_arrays(paths_list):
    array_list = []
    for path in paths_list:
        print("Processing : ", path)
        src = rasterio.open(path)
        all_bands_arr = src.read(out_dtype=np.float32)
        for bands in all_bands_arr:
            array_list.append(bands)
        src.close()
    return array_list  


# ############# 
# Unknown stuff 
# #############

def test_classes_python():
    all_classes =  [0, 1, 2, 3, 4, 5, 6]
    predicted_classes = [0, 1, 2, 5]
    values = [44.44, 22.33, 443.55, 778.45]
    all_true_values = []

    iter_num = 0
    for class_num in all_classes:
        if iter_num < len(predicted_classes):
            if class_num == predicted_classes[iter_num]:
                all_true_values.append(values[iter_num])
                iter_num += 1
            elif class_num not in predicted_classes:
                all_true_values.append(0)
        else:
            all_true_values.append(0)
        print(all_true_values)  
from libtiff import TIFF


def read_bgrnir_tif_as_bgr(image_path):
    """Reads .tif BGR-NIR image as a BGR array
    """
    tif_file = TIFF.open(image_path)

    # Remove NIR channel
    image = tif_file.read_image()[:, :, :3]
    tif_file.close()
    return image

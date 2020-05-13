from skimage.measure import compare_ssim, compare_psnr, compare_nrmse


def calc_nrmse(img_ref, img_test):
    return compare_nrmse(img_ref, img_test)

def calc_ssim(img_ref, img_test):
    return compare_ssim(img_ref, img_test, multichannel=True)

def calc_psnr(img_ref, img_test):
    return compare_psnr(img_ref, img_test, data_range=255)


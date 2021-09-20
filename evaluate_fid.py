import numpy as np

from scipy.linalg import sqrtm
from skimage.transform import resize
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input


class Evaluation:
    def __init__(self, dataset, model):
        self.inception_model = InceptionV3(
            include_top=False, pooling='avg', input_shape=(299, 299, 3), weights=None)
        self.dataset = dataset
        self.model = model

    # calculate frechet inception distance
    def calculate_fid(self, images1, images2):
        local_weights_file = "static/model/201710370311030/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
        self.inception_model.load_weights(local_weights_file)

        # # calculate activations
        act1 = self.inception_model.predict(images1)
        act2 = self.inception_model.predict(images2)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        # calculate sum squared differencestyle="width: 10%;" between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def scale_images(self, images, new_shape):
        images_list = list()
        for image in images:
            # resize with nearest neighbor interpolation
            new_image = resize(image, new_shape, 0)
            # store
            images_list.append(new_image)
        return np.asarray(images_list)

    def split_img(self, imgs, size=32):
        list_img = []
        for img in imgs:
            for r in range(0, img.shape[0], size):
                for c in range(0, img.shape[1], size):
                    list_img.append(img[r:r+size, c:c+size, :])
        img_arr = np.array(
            [img for img in list_img if img.shape == (size, size, 3)])
        return img_arr

    def fid_local(self, patchA, patchB, batik_generate):
        patch = np.concatenate([patchA, patchB], axis=0)
        img_gen = self.split_img(batik_generate)

        img_gen = self.scale_images(img_gen, (299, 299, 3))
        patch = self.scale_images(patch, (299, 299, 3))

        # pre-process images
        img_gen = preprocess_input(img_gen)
        patch = preprocess_input(patch)

        fid = self.calculate_fid(img_gen, patch)
        return fid

    def fid_global(self, id_A, id_B, batik_generate):
        real = np.array([img[2][0] for index, img in enumerate(
            list(self.dataset.as_numpy_iterator()))])  # Real Batik
        # ID = list(dict.fromkeys(ID_A + ID_B))  # ID Select
        # real_select = np.array([real[index] for index in ID]) # Real Batik Select from ID

        batik_generate = self.scale_images(batik_generate, (299, 299, 3))
        real_select = self.scale_images(real, (299, 299, 3))

        # pre-process images
        batik_generate = preprocess_input(batik_generate)
        real_select = preprocess_input(real_select)

        # calculate fid
        fid = self.calculate_fid(batik_generate, real_select)
        return fid

    def fid(self, patchA, patchB, batik_generate):
        fid_global = self.fid_global(patchA, patchB, batik_generate)
        fid_local = self.fid_local(patchA, patchB, batik_generate)
        return fid_local, fid_global
        # print('FID Global: %.3f' % fid_global)
        # print('FID Local: %.3f' % fid_local)

import keras
from keras_applications import vgg16

from tqdm import tqdm
from scipy.misc import imread, imresize
import numpy as np
from glob import glob
from skimage.filters import gabor
from skimage.morphology import skeletonize
from skimage.util import invert
from skimage.filters import threshold_otsu

import os

class BatchGenerator_Matching:

    def __init__(self, path, imsize, keras_mode=False):


        self.imsize = imsize
        self.keras_mode = keras_mode

        self.images, self.ids = self.parse_data(path, keras_mode)
        self.sample_ids = list(set([x[1:] for x in self.ids]))

        self.sample_ids_train = self.sample_ids[:32]
        self.sample_ids_val = self.sample_ids[32:64]
        self.pre = vgg16.VGG16(include_top=False, input_shape=(imsize,imsize,3), pooling='avg')

    def parse_data(self, path, keras_mode=False):
        print(keras_mode)

        file_list = glob(os.path.join(path,'figs_0','*.png'))

        ids = list(set([x[:-4].split(os.sep)[-1] for x in file_list]))
        # ids.remove('Thumb')

        images = []

        for id in tqdm(ids):
            image_path = [x for x in file_list if x.find(id) > -1 and x.endswith('png')][0]

            if keras_mode:
                img = imread(image_path, mode='RGB')
                img = np.expand_dims(img, axis=0)
                img = vgg16.preprocess_input(img)

            else:
                img = imread(image_path)
                img = img / 255
                img = img.reshape([self.imsize, self.imsize, 1])

            images.append(img)

        return images, ids


    def generate_triplet_batch(self, batch_size, candidate_ids, augment, keras_mode=False):

        batch = []

        for _ in range(batch_size):

            anchor_ids = np.random.choice(candidate_ids, 2)
            anchor_indeces = [self.ids.index('f' + anchor_id) for anchor_id in anchor_ids]
            pos_indeces = [self.ids.index('s' + anchor_id) for anchor_id in anchor_ids]
            if np.random.rand() < .5:
                anchor_indeces, pos_indeces = pos_indeces, anchor_indeces

            neg_candidate_ids = ['f' + x for x in candidate_ids if x != anchor_ids]+\
                                ['s' + x for x in candidate_ids if x != anchor_ids]
            neg_ids = np.random.choice(neg_candidate_ids, 2)
            neg_indeces = [self.ids.index(neg_id) for neg_id in neg_ids]

            anchor_img, pos_img, neg_img = self.images[anchor_indeces[0]], self.images[pos_indeces[0]], self.images[neg_indeces[0]]

            if augment:
                if np.random.rand() > 2:
                    anchor_img_p, pos_img_p, neg_img_p = self.images[anchor_indeces[1]], self.images[pos_indeces[1]], self.images[neg_indeces[1]]
                    anchor_img = np.concatenate([anchor_img[:256,:256,:]], anchor_img_p[256:,256:,:])
                    pos_img = np.concatenate([pos_img[:256,:256,:]], pos_img_p[256:,256:,:])
                    neg_img = np.concatenate([neg_img[:256,:256,:]], neg_img_p[256:,256:,:])

                if np.random.rand() > .5:
                    anchor_img, pos_img, neg_img = np.fliplr(anchor_img), np.fliplr(pos_img), np.fliplr(neg_img)

            if keras_mode:
                anchor_img = self.pre.predict(anchor_img)
                pos_img = self.pre.predict(pos_img)
                neg_img = self.pre.predict(neg_img)

                triplet = np.concatenate([anchor_img, pos_img, neg_img], axis=0)

            else:
                triplet = np.concatenate([anchor_img, pos_img, neg_img], axis=2)

            batch.append(triplet)

        return np.array(batch)


    def generate_duo_batch_with_labels(self, batch_size, candidate_ids, augment, keras_mode=False):

        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            # anchor_id = np.random.choice(candidate_ids)

            anchor_ids = np.random.choice(candidate_ids, 2)
            anchor_indeces = [self.ids.index('f' + anchor_id) for anchor_id in anchor_ids]

            # partner_index = self.ids.index('s' + anchor_id)
            partner_indeces = [self.ids.index('s' + anchor_id) for anchor_id in anchor_ids]


            # pos case
            if np.random.rand() < .5:
                if np.random.rand() < .5:
                    anchor_indeces, partner_indeces = partner_indeces, anchor_indeces
                label = 1

            # neg case
            else:
                neg_candidate_ids = ['f' + x for x in candidate_ids if np.all(x != anchor_ids)] + \
                                    ['s' + x for x in candidate_ids if np.all(x != anchor_ids)]

                neg_ids = np.random.choice(neg_candidate_ids, 2)
                partner_indeces = [self.ids.index(neg_id) for neg_id in neg_ids]
                label = 0


            anchor_img = self.images[anchor_indeces[0]]
            partner_img = self.images[partner_indeces[0]]

            if augment:
                if np.random.rand() > 2:
                    anchor_img_p, partner_img_p = self.images[anchor_indeces[1]], self.images[partner_indeces[1]]
                    anchor_img = np.concatenate([anchor_img[:256, :256, :]], anchor_img_p[256:, 256:, :])
                    partner_img = np.concatenate([partner_img[:256, :256, :]], partner_img_p[256:, 256:, :])

                if np.random.rand() > .5:
                    anchor_img, partner_img = np.fliplr(anchor_img), np.fliplr(partner_img)

                #anchor_img += np.random.normal(0, .1, size=anchor_img.shape)
                #partner_img += np.random.normal(0, .1, size=partner_img.shape)

            if keras_mode:
                anchor_img = self.pre.predict(anchor_img)
                partner_img = self.pre.predict(partner_img)

                duo = np.concatenate([anchor_img, partner_img], axis=0)

            else:
                duo = np.concatenate([anchor_img, partner_img, partner_img], axis=2)

            x_batch.append(duo)
            y_batch.append(label)

        return np.array(x_batch), np.array(y_batch).reshape(batch_size, 1)


    def generate_train_triplets(self, batch_size, augment=True):

        return self.generate_triplet_batch(batch_size, self.sample_ids_train, augment)

    def generate_val_triplets(self, batch_size, augment=False):

        return self.generate_triplet_batch(batch_size, self.sample_ids_val, augment)

    def generate_train_duos(self, batch_size, augment=True):

        return self.generate_duo_batch_with_labels(batch_size, self.sample_ids_train, augment)

    def generate_val_duos(self, batch_size, augment=False):

        return self.generate_duo_batch_with_labels(batch_size, self.sample_ids_val, augment)
from tqdm import tqdm
from scipy.misc import imread, imresize
import numpy as np
from glob import glob


class BatchGenerator_Classification:


    def __init__(self, path, imsize):

        self.imsize = imsize
        self.images, self.labels = self.parse_data(path)
        self.images_train, self.labels_train = self.images[:1600], self.labels[:1600]
        self.images_val, self.labels_val = self.images[1600:], self.labels[1600:]


    def parse_data(self, path):


        file_list = glob(path + '/*' + '/*')

        ids = list(set([x[:-4].split('/')[-1] for x in file_list]))
        ids.remove('Thumb')

        images = []
        labels = []

        for id in tqdm(ids):
            image_path = [x for x in file_list if x.find(id) > -1 and x.endswith('png')][0]
            label_path = [x for x in file_list if x.find(id) > -1 and x.endswith('txt')][0]

            img = imread(image_path)
            label_file = open(label_path)
            for line in label_file.readlines():
                if line.startswith('Class'):
                    label = line[7]

            img = img / 255
            images.append(img)
            labels.append(label)

        # Tokenize labels
        tokens = list(range(len(set(labels))))
        self.label_dict = {key: value for key, value in zip(set(labels), tokens)}
        labels_tokenized = [self.label_dict[x] for x in labels]
        n_values = np.max(tokens) + 1
        labels_one_hot = np.eye(n_values)[labels_tokenized]

        return images, labels_one_hot


    def generate_batch(self, batch_size, images, labels):

        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            index = np.random.choice(range(len(images)))
            x_batch.append(images[index])
            y_batch.append(labels[index])

        return np.array(x_batch).reshape(batch_size, self.imsize, self.imsize, 1), np.array(y_batch)


    def generate_train_batch(self, batch_size):

        return self.generate_batch(batch_size, self.images_train, self.labels_train)


    def generate_val_batch(self, batch_size):

        return self.generate_batch(batch_size, self.images_val, self.labels_val)


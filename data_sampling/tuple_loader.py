import imageio
import numpy as np
import pandas as pd

class BaseTupleLoader:
    def __init__(self,args):
        '''
        Given a csv file and an absolute dataset path, this creates a list of images' paths (absolute paths).
        CSV files contains rows that denote labels and images within every split.

        :param args: Python dictionary with the following keys
            - csv_file
            - db_path
            - shuffle: Boolean; shuffle the list of created images or not
        '''
        csv_file = args['csv_file']
        db_path = args['db_path']
        self.data_df = pd.read_csv(db_path + csv_file)

        shuffle_data = args['shuffle']
        if shuffle_data:
            self.data_permutation = np.random.permutation(self.data_df.shape[0])
        else:
            self.data_permutation = list(range(self.data_df.shape[0]))

    def load_img_sync(self,img_path):
        img = imageio.imread(img_path);
        return img

    def load_img_async(self, img_path):
        try:
            return self.load_img_sync(img_path)
        except:
            return (None)



    def imgs_and_lbls(self,repeat=False):
        """
        Load images' paths and int32 labels
        :param repeat: This is similar to TF.data.Dataset repeat. I use TF dataset repeat and no longer user this params.
        So its default is False

        :return: a list of images' paths and their corresponding int32 labels
        """

        imgs = self.data_df
        ## Faster way to read data
        images = imgs['file_name'].tolist()
        lbls = imgs['label'].tolist()
        for img_idx in range(imgs.shape[0]):
            images[img_idx] = self.img_path + images[img_idx]
            lbls[img_idx] = self.lbl2idx_dict[lbls[img_idx]]

        if repeat:
            return self.repeat_dataset(images, lbls)
        else:
            return images, lbls

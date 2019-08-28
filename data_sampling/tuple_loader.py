import imageio
import numpy as np
import pandas as pd

class BaseTupleLoader:
    def __init__(self,args):
        # print('Base Tuple loader')
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
            #print(img_path)
            return self.load_img_sync(img_path)
        except:
            return (None)



    def imgs_and_lbls(self,repeat=False):

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


from tensorflow.python import pywrap_tensorflow
import numpy as np
import h5py as h5
def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.iteritems():
        print("    %s: %s" % (key, val))


def print_tensors_in_h5(file_name):
    f = h5.File(file_name)  # create a file object print f.keys() # print file contents
    f.visititems(print_attrs)
    # keys = [key for key in f.keys()]
    # for key in keys:
    #     if(type(f[key]) == h5._hl.group.Group): ## This is a group
    #         group = f[key];
    #         items = f[key].items()



def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors,
                                     all_tensor_names):
    """Prints tensors in a checkpoint file.
    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.
    Args:
      file_name: Name of the checkpoint file.
      tensor_name: Name of the tensor in the checkpoint file to print.
      all_tensors: Boolean indicating whether to print all tensors.
      all_tensor_names: Boolean indicating whether to print all tensor names.
    """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors or all_tensor_names:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                print("tensor_name: ", key, reader.get_tensor(key).shape,np.mean(reader.get_tensor(key)))
                # if all_tensors:
                #      print(reader.get_tensor(key))
        elif not tensor_name:
            print(reader.debug_string().decode("utf-8"))
        else:
            print("tensor_name: ", tensor_name)
            print(reader.get_tensor(tensor_name))
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")
        if ("Data loss" in str(e) and
                (any([e in file_name for e in [".index", ".meta", ".data"]]))):
            proposed_file = ".".join(file_name.split(".")[0:-1])
            v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.  Try removing the '.' and extension.  Try:
inspect checkpoint --file_name = {}"""

        print(v2_file_error_template.format(proposed_file))


if __name__ == '__main__':
    #print_tensors_in_h5('/Users/ahmedtaha/Documents/Models/densenet161/densenet161_weights_tf.h5')
    # print_tensors_in_checkpoint_file('/Users/ahmedtaha/Documents/Model/resnet_v1_50/resnet_v1_50.ckpt',tensor_name='resnet_v1',all_tensors=True,all_tensor_names=True)
    print_tensors_in_checkpoint_file('/Users/ahmedtaha/Downloads/faster_rcnn_resnet50_coco_2018_01_28/model.ckpt',
                                     tensor_name='', all_tensors=True, all_tensor_names=True)